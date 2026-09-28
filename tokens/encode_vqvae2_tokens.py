"""Generate the VQ-VAE-2 token datasets for the three CEU datasets (KH, RC, RKH).

Writes .nc files in the SAME schema as the Phaedra token files so the unmodified
operator pipeline trains on them:
  <var>_amp   = TOP tokens (vocab 4096), native 16x16, replicated 2x2 onto 32x32
  <var>_morph = BOTTOM tokens (vocab 16384), native 32x32
  dims (member, time=21, token_x=32, token_y=32); attrs mirror the Phaedra token
  files (amplitude_codebook_size=4096, morphology_codebook_size=16384,
  morphology_offset=0, split ranges, time_indices=[0,...,20]).

The 2x2 replication is exact at generation (each top token covers a 2x2
footprint of bottom positions); at decode time predicted top tokens are
collapsed back to 16x16 by a per-2x2-block majority vote
(evaluation.eval_downstream.ArchDualSequentialVQ2Runner).

Also prints the VQ-VAE2 reconstruction floor (decode(encode(x)) vs x, test
members) for the tokenizer-comparison table.

Usage:
  python -m tokens.encode_vqvae2_tokens --datasets kh rc rkh
  python -m tokens.encode_vqvae2_tokens --datasets kh --max-members 2  # smoke
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf

from fno_operator.src.dataset import read_fields
from hub.trainers.test import _relative_l1
from hub.utils.phaedra_decoder import _apply_ema, _load_state_dict, _normalize_state_dict
from hub.utils.runtime import configure_torch, get_device, portable_path
from tokenizer import config_path as _tok_config

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "evaluation" / "eval_registry.yaml"
OUT_DIR = Path(os.environ.get("PHAEDRA_DATA_ROOT", ".")) / "tokens/vqvae2"
AE_CONFIG = str(_tok_config("AE_VQVAE2"))
AE_WEIGHTS = str(Path(os.environ.get("PHAEDRA_OUTPUT_ROOT", ".")) / "tokenizers" / "vqvae2")
# All 21 source timesteps: the hub CEUKHTokenDataset indexes the time axis by
# RAW timestep (0..20), exactly like Phaedra_tokens/. Do not subset.
TIME_INDICES = list(range(21))
TOP_VOCAB = 4096
BOT_VOCAB = 16384
DATASET_FILE = {
    "kh": "CEU2D_KelvinHelmholtzTokens.nc",
    "rc": "CEU2D_RiemannCurvedTokens.nc",
    "rkh": "CEU2D_RiemannKelvinHelmholtzTokens.nc",
}


def load_vqvae2(device: torch.device, use_ema: bool = True, weights: str | None = None):
    from tokenizer.systems import VQVAE2AESystem

    weights = weights or AE_WEIGHTS
    task = VQVAE2AESystem(OmegaConf.load(AE_CONFIG))
    from hub.utils.weights import is_released
    released = is_released(weights)           # released safetensors already hold the EMA weights
    state = _normalize_state_dict(_load_state_dict(Path(weights)))
    task.model.load_state_dict(state, strict=released)
    if use_ema and not released:
        _apply_ema(task, Path(weights) / "ema.pt", device)
    task.model.to(device)
    task.model.eval()
    n = sum(p.numel() for p in task.model.parameters())
    print(f"[vq2] VQ-VAE2 loaded ({n / 1e6:.1f}M, ema={use_ema})", flush=True)
    return task.model


@torch.no_grad()
def encode_tokens(model, fields_norm: np.ndarray, device):
    """[N,128,128] -> (top [N,8,8] int64, bottom [N,16,16] int64)."""
    x = torch.from_numpy(fields_norm).float().unsqueeze(1).to(device)
    _, _, (tok_t, tok_b), _ = model(x, mode="encode")
    N = x.shape[0]
    tok_t = tok_t.reshape(N, -1)
    tok_b = tok_b.reshape(N, -1)
    st = int(tok_t.shape[1] ** 0.5)
    sb = int(tok_b.shape[1] ** 0.5)
    return (tok_t.reshape(N, st, st).long().cpu().numpy(),
            tok_b.reshape(N, sb, sb).long().cpu().numpy())


@torch.no_grad()
def decode_tokens_pair(model, top: np.ndarray, bottom: np.ndarray, device) -> np.ndarray:
    """(top [N,8,8], bottom [N,16,16]) int -> fields [N,128,128] normalized."""
    tt = torch.from_numpy(top).long().to(device)
    tb = torch.from_numpy(bottom).long().to(device)
    et = model.quantizer_t.embedding(tt.reshape(tt.shape[0], -1))
    eb = model.quantizer_b.embedding(tb.reshape(tb.shape[0], -1))
    st, sb = tt.shape[-1], tb.shape[-1]
    quant_t = et.reshape(tt.shape[0], st, st, -1).permute(0, 3, 1, 2).contiguous()
    quant_b = eb.reshape(tb.shape[0], sb, sb, -1).permute(0, 3, 1, 2).contiguous()
    out = model((quant_t, quant_b), mode="decode")
    return out.detach().float().cpu().numpy()[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["kh", "rc", "rkh"])
    ap.add_argument("--ae-weights", default=AE_WEIGHTS, help="trained AE checkpoint dir (pytorch_model.bin + ema.pt)")
    ap.add_argument("--n-val", type=int, default=120, help="validation members (taken before the test block)")
    ap.add_argument("--n-test", type=int, default=240, help="test members (last members of the file)")
    ap.add_argument("--max-members", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--roundtrip-members", type=int, default=8)
    ap.add_argument("--start-member", type=int, default=0,
                    help="Resume: open existing .nc r+ and continue from this member")
    ap.add_argument("--end-member", type=int, default=None,
                    help="Exclusive end member (sharded generation)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Write to a shard file <name><suffix>.nc instead of the main file")
    args = ap.parse_args()

    configure_torch(tf32=True)
    device = get_device()
    reg = OmegaConf.to_container(OmegaConf.load(REGISTRY), resolve=True)
    model = load_vqvae2(device, weights=args.ae_weights)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

    for ds_name in args.datasets:
        ds_cfg = reg["datasets"][ds_name]
        variables = list(ds_cfg["variables"])
        norm = {k: (float(v["mean"]), float(v["std"])) for k, v in ds_cfg["normalization"].items()}
        src = nc.Dataset(ds_cfg["source_fields"], "r")
        n_total = int(src.dimensions["member"].size)
        n_members = min(args.max_members or n_total, n_total)
        fname = DATASET_FILE[ds_name]
        if args.out_suffix:
            fname = fname.replace(".nc", f"{args.out_suffix}.nc")
        out_path = OUT_DIR / fname
        print(f"[vq2] {ds_name}: {n_members} members -> {out_path}", flush=True)

        end_member = int(args.end_member) if args.end_member is not None else n_members
        resume = int(args.start_member) > 0 and out_path.exists() and not args.out_suffix
        out = nc.Dataset(out_path, "r+" if resume else "w")
        if args.out_suffix:
            out.setncattr("shard_start", int(args.start_member))
            out.setncattr("shard_end", int(end_member))
        if resume:
            print(f"[vq2] {ds_name}: RESUME at member {args.start_member}", flush=True)
        if not resume:
            out.createDimension("member", n_members)
        if not resume:
            out.createDimension("time", len(TIME_INDICES))
            out.createDimension("token_x", 32)
            out.createDimension("token_y", 32)
        for var in (variables if not resume else []):
            out.createVariable(f"{var}_amp", "i4", ("member", "time", "token_x", "token_y"),
                               zlib=True, complevel=1)
            out.createVariable(f"{var}_morph", "i4", ("member", "time", "token_x", "token_y"),
                               zlib=True, complevel=1)
        if not resume:
            mvar = out.createVariable("member", "i8", ("member",))
            mvar[:] = np.arange(n_members)
        if not resume:
            out.setncattr("time_indices", np.array(TIME_INDICES, dtype=np.int32))
            out.setncattr("variables", ",".join(variables))
            out.setncattr("amplitude_codebook_size", TOP_VOCAB)      # amp stream = TOP
            out.setncattr("morphology_codebook_size", BOT_VOCAB)     # morph stream = BOTTOM
            out.setncattr("morphology_offset", 0)
            out.setncattr("top_grid", 16)
            out.setncattr("replication", "top tokens replicated 2x2 onto 32x32 amp grid")
            out.setncattr("source_dataset", Path(ds_cfg["source_fields"]).name)
            out.setncattr("model_name", "AE_VQVAE2")
            out.setncattr("model_path", portable_path(AE_WEIGHTS))
            out.setncattr("git_commit", git)
            _tr = n_total - args.n_val - args.n_test
            out.setncattr("split_train_range", f"0:{_tr}")
            out.setncattr("split_val_range", f"{_tr}:{_tr + args.n_val}")
            out.setncattr("split_test_range", f"{_tr + args.n_val}:{n_total}")

        t0 = time.time()
        B_members = max(1, args.batch // (len(TIME_INDICES) * len(variables)))
        for lo in range(int(args.start_member), end_member, B_members):
            hi = min(lo + B_members, end_member)
            fields = np.zeros(((hi - lo), len(TIME_INDICES), len(variables), 128, 128),
                              dtype=np.float32)
            for j, mi in enumerate(range(lo, hi)):
                # One bulk read per (member, var): 21x fewer netCDF calls than
                # per-(member, t) reads -- the generation was IO-bound.
                for vi, var in enumerate(variables):
                    m, s = norm[var]
                    arr = np.asarray(src.variables[var][mi], dtype=np.float32)
                    fields[j, :, vi] = (arr[TIME_INDICES] - m) / s
            K = fields.shape[0]
            flat = fields.reshape(K * len(TIME_INDICES) * len(variables), 128, 128)
            top, bot = encode_tokens(model, flat, device)
            assert bot.shape[-1] == 32 and top.shape[-1] == 16, \
                f'unexpected VQ2 grids top={top.shape} bot={bot.shape}'
            top = top.reshape(K, len(TIME_INDICES), len(variables), *top.shape[1:])
            bot = bot.reshape(K, len(TIME_INDICES), len(variables), *bot.shape[1:])
            # replicate top (16x16) -> (32x32)
            top32 = np.repeat(np.repeat(top, 2, axis=-2), 2, axis=-1)
            for vi, var in enumerate(variables):
                out.variables[f"{var}_amp"][lo:hi] = top32[:, :, vi].astype(np.int32)
                out.variables[f"{var}_morph"][lo:hi] = bot[:, :, vi].astype(np.int32)
            if lo % (B_members * 50) == 0:
                print(f"[vq2] {ds_name}: member {lo}/{n_members} "
                      f"({time.time() - t0:.0f}s) top_grid={top.shape[-1]} "
                      f"bot_grid={bot.shape[-1]}", flush=True)
        out.close()
        print(f"[vq2] {ds_name}: wrote {out_path} in {time.time() - t0:.0f}s", flush=True)

        if args.out_suffix:
            src.close()
            continue
        # round-trip floor on test members (t=14)
        n_rt = min(args.roundtrip_members, n_members)
        start = (n_total - args.n_test) if n_members == n_total else 0
        tok = nc.Dataset(out_path, "r")
        errs = {v: [] for v in variables}
        for mi in range(start, start + n_rt):
            for vi, var in enumerate(variables):
                amp32 = np.asarray(tok.variables[f"{var}_amp"][mi, 14], dtype=np.int64)
                bot = np.asarray(tok.variables[f"{var}_morph"][mi, 14], dtype=np.int64)
                top16 = amp32[::2, ::2]  # exact inverse of the replication
                rec = decode_tokens_pair(model, top16[None], bot[None], device)[0]
                m, s = norm[var]
                true_raw = read_fields(src, member_index=mi, time_idx=14, variables=[var])[0]
                errs[var].append(_relative_l1(rec * s + m, true_raw))
        tok.close()
        means = {v: float(np.mean(e)) for v, e in errs.items()}
        print(f"[vq2] {ds_name}: recon floor (n={n_rt}, t=14): {means} "
              f"avg={np.mean(list(means.values())):.5f}", flush=True)
        src.close()


if __name__ == "__main__":
    main()
