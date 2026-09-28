"""One-off: encode source CFD fields through the frozen continuous AE.

Writes per-dataset latent archives that mirror the token .nc layout so the
continuous-latent transformer trains with the same all2all pair logic:

  latents float16, dims (member=10000, time=8, var=4, ch=8, lh=32, lw=32)
  where time axis holds source timesteps {0,2,...,14} (attr `time_indices`).

Normalization: fields are normalized per variable with the SAME (mean, std)
used everywhere else in the pipeline (evaluation/eval_registry.yaml) BEFORE
encoding; the AE was trained on normalized fields, and decode() returns
normalized fields (denormalize for physical-space metrics).

Also prints the continuous-AE round-trip reconstruction floor (relL1 of
decode(encode(x)) vs x) over the test split's first N members — the direct
counterpart of the Phaedra tokenizer floor in Table 3.

Usage:
  python -m tokens.encode_continuous_latents --datasets kh rc rkh
  python -m tokens.encode_continuous_latents --datasets kh --max-members 2  # smoke
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
OUT_DIR = Path(os.environ.get("PHAEDRA_DATA_ROOT", ".")) / "latents/continuous"
AE_CONFIG = str(_tok_config("AE_Continuous"))
AE_WEIGHTS = str(Path(os.environ.get("PHAEDRA_OUTPUT_ROOT", ".")) / "tokenizers" / "continuous")
TIME_INDICES = [0, 2, 4, 6, 8, 10, 12, 14]
DATASET_FILE = {
    "kh": "CEU2D_KelvinHelmholtzLatents.nc",
    "rc": "CEU2D_RiemannCurvedLatents.nc",
    "rkh": "CEU2D_RiemannKelvinHelmholtzLatents.nc",
}


def load_continuous_ae(device: torch.device, use_ema: bool = True, weights: str | None = None):
    from tokenizer.systems import ContinuousAESystem

    weights = weights or AE_WEIGHTS
    task = ContinuousAESystem(OmegaConf.load(AE_CONFIG))
    from hub.utils.weights import is_released
    released = is_released(weights)           # released safetensors already hold the EMA weights
    state = _normalize_state_dict(_load_state_dict(Path(weights)))
    task.model.load_state_dict(state, strict=released)
    if use_ema and not released:
        _apply_ema(task, Path(weights) / "ema.pt", device)
    task.model.to(device)
    task.model.eval()
    n = sum(p.numel() for p in task.model.parameters())
    print(f"[enc] continuous AE loaded ({n / 1e6:.1f}M params, ema={use_ema})", flush=True)
    return task.model


@torch.no_grad()
def encode_batch(model, fields_norm: np.ndarray, device) -> np.ndarray:
    """fields_norm [N, 128, 128] -> latents [N, 8, 32, 32] float16."""
    x = torch.from_numpy(fields_norm).float().unsqueeze(1).to(device)  # [N,1,128,128]
    z = model(x, mode="encode")
    return z.detach().to(torch.float16).cpu().numpy()


@torch.no_grad()
def decode_batch(model, z: np.ndarray, device) -> np.ndarray:
    zt = torch.from_numpy(z).float().to(device)
    x = model(zt, mode="decode")
    return x.detach().float().cpu().numpy()[:, 0]  # [N,128,128]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["kh", "rc", "rkh"])
    ap.add_argument("--ae-weights", default=AE_WEIGHTS, help="trained AE checkpoint dir (pytorch_model.bin + ema.pt)")
    ap.add_argument("--n-val", type=int, default=120, help="validation members (taken before the test block)")
    ap.add_argument("--n-test", type=int, default=240, help="test members (last members of the file)")
    ap.add_argument("--max-members", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--roundtrip-members", type=int, default=8,
                    help="test-split members for the recon-floor check")
    args = ap.parse_args()

    configure_torch(tf32=True)
    device = get_device()
    reg = OmegaConf.to_container(OmegaConf.load(REGISTRY), resolve=True)
    model = load_continuous_ae(device, weights=args.ae_weights)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    git_commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()

    for ds_name in args.datasets:
        ds_cfg = reg["datasets"][ds_name]
        variables = list(ds_cfg["variables"])
        norm = {k: (float(v["mean"]), float(v["std"])) for k, v in ds_cfg["normalization"].items()}
        src = nc.Dataset(ds_cfg["source_fields"], "r")
        n_members_total = int(src.dimensions["member"].size)
        n_members = min(args.max_members or n_members_total, n_members_total)
        out_path = OUT_DIR / DATASET_FILE[ds_name]
        print(f"[enc] {ds_name}: {n_members} members x {len(TIME_INDICES)} t x "
              f"{len(variables)} vars -> {out_path}", flush=True)

        out = nc.Dataset(out_path, "w")
        out.createDimension("member", n_members)
        out.createDimension("time", len(TIME_INDICES))
        out.createDimension("var", len(variables))
        out.createDimension("ch", 8)
        out.createDimension("lh", 32)
        out.createDimension("lw", 32)
        v_lat = out.createVariable("latents", "f4", ("member", "time", "var", "ch", "lh", "lw"),
                                   zlib=True, complevel=1,
                                   chunksizes=(1, len(TIME_INDICES), len(variables), 8, 32, 32))
        # (f4 on disk with zlib compresses well; fp16 write support in netCDF4 is
        # spotty across backends — f4+zlib lands near the fp16 estimate.)
        out.setncattr("time_indices", np.array(TIME_INDICES, dtype=np.int32))
        out.setncattr("variables", ",".join(variables))
        out.setncattr("source_dataset", Path(ds_cfg["source_fields"]).name)
        out.setncattr("ae_weights", portable_path(AE_WEIGHTS))
        out.setncattr("ae_config", portable_path(AE_CONFIG))
        out.setncattr("git_commit", git_commit)
        out.setncattr("normalization_mean", np.array([norm[v][0] for v in variables]))
        out.setncattr("normalization_std", np.array([norm[v][1] for v in variables]))
        for attr in ("split_train_range", "split_val_range", "split_test_range"):
            # splits defined on member IDs; copy from the token file convention
            _tr = n_members_total - args.n_val - args.n_test
            out.setncattr(attr, {"split_train_range": f"0:{_tr}",
                                 "split_val_range": f"{_tr}:{_tr + args.n_val}",
                                 "split_test_range": f"{_tr + args.n_val}:{n_members_total}"}[attr])

        t0 = time.time()
        buf_members, buf_fields = [], []

        def flush():
            if not buf_members:
                return
            stacked = np.stack(buf_fields)  # [K, T, V, 128, 128]
            K, T, V = stacked.shape[:3]
            flat = stacked.reshape(K * T * V, 128, 128)
            zs = []
            for i in range(0, len(flat), args.batch):
                zs.append(encode_batch(model, flat[i:i + args.batch], device))
            z = np.concatenate(zs).reshape(K, T, V, 8, 32, 32)
            for j, mi in enumerate(buf_members):
                v_lat[mi] = z[j].astype(np.float32)
            buf_members.clear()
            buf_fields.clear()

        for mi in range(n_members):
            fields = np.zeros((len(TIME_INDICES), len(variables), 128, 128), dtype=np.float32)
            for ti, t in enumerate(TIME_INDICES):
                raw = read_fields(src, member_index=mi, time_idx=t, variables=variables)
                for vi, var in enumerate(variables):
                    m, s = norm[var]
                    fields[ti, vi] = (raw[vi] - m) / s
            buf_members.append(mi)
            buf_fields.append(fields)
            if len(buf_members) >= 8:
                flush()
            if mi % 250 == 0:
                el = time.time() - t0
                print(f"[enc] {ds_name}: member {mi + 1}/{n_members} ({el:.0f}s)", flush=True)
        flush()
        out.close()
        print(f"[enc] {ds_name}: wrote {out_path} in {time.time() - t0:.0f}s", flush=True)

        # ---- round-trip reconstruction floor on test members ----
        lat = nc.Dataset(out_path, "r")
        n_rt = min(args.roundtrip_members, n_members)
        start_idx = (n_members_total - args.n_test) if n_members == n_members_total else 0
        errs = {v: [] for v in variables}
        for mi in range(start_idx, start_idx + n_rt):
            z = np.asarray(lat.variables["latents"][mi, -1], dtype=np.float32)  # t=14
            rec = decode_batch(model, z, device)                                # [V,128,128] norm
            true_raw = read_fields(src, member_index=mi, time_idx=14, variables=variables)
            for vi, var in enumerate(variables):
                m, s = norm[var]
                errs[var].append(_relative_l1(rec[vi] * s + m, true_raw[vi]))
        lat.close()
        means = {v: float(np.mean(e)) for v, e in errs.items()}
        print(f"[enc] {ds_name}: continuous-AE recon floor (n={n_rt}, t=14): "
              f"{means}  avg={np.mean(list(means.values())):.5f}", flush=True)
        src.close()


if __name__ == "__main__":
    main()
