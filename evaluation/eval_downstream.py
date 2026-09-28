"""Evaluation harness of the paper: per-trajectory, per-timestep relative L1.

For every registered model x its dataset: iterate the 240 test trajectories,
predict with the chosen strategy (--mode: direct t_in=0 -> t_out for every even
t_out in {2,...,14}; 2-step or 6-6-2 autoregressive rollouts), decode to physical
space, denormalize, and record relative L1 per variable. Output: tidy CSV rows
    model,dataset,traj_id,var,t,relL1
in $PHAEDRA_OUTPUT_ROOT/evaluation/tables/per_sample_relL1[_<mode>].csv (atomic per
model: existing rows for a model+dataset are replaced, others preserved).

Families (evaluation/eval_registry.yaml):
  hub_phaedra / hub_fsq     -- token transformers on Phaedra / FSQ tokens
  arch_dual_sequential_vq2  -- transformer on VQ-VAE-2 tokens
  continuous                -- transformer on continuous-AE latents
  fno / cno / vit           -- physical-space baselines

Usage:
  python -m evaluation.eval_downstream --models phaedra_38m_kh fsq_38m_kh
  python -m evaluation.eval_downstream --all
  python -m evaluation.eval_downstream --all --max-members 2 --timesteps 2 14  # smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf

from fno_operator.src.dataset import normalize_fields, read_fields
from hub.config import load_config as hub_load_config
from hub.models import build_model as hub_build_model
from hub.trainers.test import (
    _clip_tokens_for_decoder_fsq,
    _clip_tokens_for_decoder_phaedra,
    _decode_normalized_fields,
    _extract_model_state,
    _load_member_tokens_fsq,
    _load_member_tokens_phaedra,
    _normalize_state_dict_keys,
    _predict_step_seq2seq_fsq,
    _predict_step_seq2seq_phaedra,
    _relative_l1,
)
from hub.utils.phaedra_decoder import load_token_decoder
from hub.utils.weights import load_weights
from hub.utils.runtime import configure_torch, get_device

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "evaluation" / "eval_registry.yaml"
EVAL_ROOT = Path(os.environ.get("PHAEDRA_OUTPUT_ROOT", ".")) / "evaluation"
OUT_CSV = EVAL_ROOT / "tables" / "per_sample_relL1.csv"
META_JSON = EVAL_ROOT / "tables" / "per_sample_relL1.meta.json"
CSV_FIELDS = ["model", "dataset", "traj_id", "var", "t", "relL1"]


# ---------------------------------------------------------------------------
# Model loaders per family. Each returns (predict_fn, meta) where
# predict_fn(start_state, t_out) -> dict var -> np.ndarray [128,128] physical.
# `start_state` is family-specific and produced by make_start_state(member).
# ---------------------------------------------------------------------------

def _payload_int(payload, key: str):
    """Best-effort int(payload[key]) for checkpoint metadata (epoch/step)."""
    if isinstance(payload, dict) and payload.get(key) is not None:
        try:
            return int(payload[key])
        except (TypeError, ValueError):
            return None
    return None


def _load_ckpt_state(path: str, device):
    """(state_dict, payload) from a released `model.safetensors`, a training
    checkpoint, or a run directory. For released weights `payload` carries only
    epoch/step (from the safetensors metadata)."""
    state, info = load_weights(path, map_location=device)
    payload = info["payload"] if info["payload"] is not None else {"epoch": info["epoch"], "step": info["step"]}
    return state, payload


def _arch_config(entry: dict, payload: dict) -> dict:
    """Config for the dual-decoder (VQ-VAE-2) builder: pickled in training
    checkpoints, taken from the repo config (`op_config`) for released weights."""
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        return payload["config"]
    from baselines.arch.config_io import load_config as _arch_load_config
    return _arch_load_config(str(REPO / entry["op_config"]))


class HubTokenRunner:
    """Token transformers on Phaedra tokens (amplitude + morphology heads) or FSQ tokens."""

    def __init__(self, entry: dict, ds_cfg: dict, decoders: dict, device):
        self.family = entry["family"]              # hub_phaedra | hub_fsq
        self.device = device
        self.variables = list(ds_cfg["variables"])
        self.norm = {k: (v["mean"], v["std"]) for k, v in ds_cfg["normalization"].items()}
        cfg = hub_load_config(str(REPO / entry["hub_config"]))
        self.model = hub_build_model(cfg).to(device)
        state, payload = _load_ckpt_state(entry["checkpoint"], device)
        res = self.model.load_state_dict(state, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"{entry['checkpoint']}: {len(res.missing_keys)} missing keys")
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.trained_epoch = _payload_int(payload, "epoch")
        self.trained_step = _payload_int(payload, "step")

        if self.family == "hub_phaedra":
            self.token_path = ds_cfg["phaedra_tokens"]
            self.decoder = load_token_decoder({**decoders["phaedra"], "enabled": True}, device=device)
            self.amp_vocab, self.morph_vocab = 1024, 8640
        else:
            self.token_path = ds_cfg["fsq_tokens"]
            self.decoder = load_token_decoder({**decoders["fsq"], "enabled": True}, device=device)
            self.fsq_vocab = 8640
        self._ds = nc.Dataset(self.token_path, "r")
        self.morph_offset = (int(self._ds.getncattr("morphology_offset"))
                             if "morphology_offset" in self._ds.ncattrs() else 0)

    def make_start_state(self, member_index: int):
        if self.family == "hub_phaedra":
            amp, morph = _load_member_tokens_phaedra(
                self._ds, member_index=member_index, time_idx=0,
                variables=self.variables, morph_offset=self.morph_offset)
            return (amp, morph)
        return _load_member_tokens_fsq(
            self._ds, member_index=member_index, time_idx=0, variables=self.variables)

    @torch.no_grad()
    def step_state(self, state, t_in: int, t_out: int):
        """One model hop: state at t_in -> predicted state at t_out."""
        if self.family == "hub_phaedra":
            amp0, morph0 = state
            pa, pm = _predict_step_seq2seq_phaedra(
                self.model, amp0, morph0, int(t_in), int(t_out), self.device)
            return _clip_tokens_for_decoder_phaedra(pa.cpu(), pm.cpu(),
                                                    self.amp_vocab, self.morph_vocab)
        pf = _predict_step_seq2seq_fsq(self.model, state, int(t_in), int(t_out), self.device)
        return _clip_tokens_for_decoder_fsq(pf.cpu(), self.fsq_vocab)

    @torch.no_grad()
    def decode_state(self, state) -> dict:
        if self.family == "hub_phaedra":
            pa, pm = state
            rec = _decode_normalized_fields(self.decoder, "phaedra",
                                            amp_tokens=pa, morph_tokens=pm)
        else:
            rec = _decode_normalized_fields(self.decoder, "fsq", fsq_tokens=state)
        return {v: rec[i] * self.norm[v][1] + self.norm[v][0]
                for i, v in enumerate(self.variables)}

    def predict_fields(self, start_state, t_out: int) -> dict:
        return self.decode_state(self.step_state(start_state, 0, t_out))

    def close(self):
        self._ds.close()


class ArchDualSequentialRunner:
    """Scaling-sweep checkpoints (dual-decoder builder in baselines.arch)."""

    def __init__(self, entry: dict, ds_cfg: dict, decoders: dict, device):
        from baselines.arch.models import BUILDERS
        self.device = device
        self.variables = list(ds_cfg["variables"])
        self.norm = {k: (v["mean"], v["std"]) for k, v in ds_cfg["normalization"].items()}
        state, payload = _load_ckpt_state(entry["checkpoint"], device)
        cfg = _arch_config(entry, payload)
        self.model = BUILDERS["dual_sequential"](cfg).to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.trained_epoch = _payload_int(payload, "epoch") if _payload_int(payload, "epoch") is not None else -1
        self.trained_step = _payload_int(payload, "step")
        self.token_path = ds_cfg["phaedra_tokens"]
        self.decoder = load_token_decoder({**decoders["phaedra"], "enabled": True}, device=device)
        self._ds = nc.Dataset(self.token_path, "r")
        self.morph_offset = (int(self._ds.getncattr("morphology_offset"))
                             if "morphology_offset" in self._ds.ncattrs() else 0)

    def make_start_state(self, member_index: int):
        return _load_member_tokens_phaedra(
            self._ds, member_index=member_index, time_idx=0,
            variables=self.variables, morph_offset=self.morph_offset)

    @torch.no_grad()
    def step_state(self, state, t_in: int, t_out: int):
        amp0, morph0 = state
        pred_amp, pred_morph = self.model.predict(
            input_amp=amp0.unsqueeze(0).to(self.device),
            input_morph=morph0.unsqueeze(0).to(self.device),
            input_time_idx=torch.tensor([int(t_in)], dtype=torch.long, device=self.device),
            output_time_idx=torch.tensor([int(t_out)], dtype=torch.long, device=self.device),
            input_var_ids=None, output_var_ids=None,
            input_var_mask=None, output_var_mask=None,
            problem_type_id=torch.zeros((1,), dtype=torch.long, device=self.device),
        )
        V, H, W = amp0.shape
        pa = pred_amp.view(1, V, H, W)[0].cpu()
        pm = pred_morph.view(1, V, H, W)[0].cpu()
        return _clip_tokens_for_decoder_phaedra(pa, pm, 1024, 8640)

    @torch.no_grad()
    def decode_state(self, state) -> dict:
        pa, pm = state
        rec = _decode_normalized_fields(self.decoder, "phaedra", amp_tokens=pa, morph_tokens=pm)
        return {v: rec[i] * self.norm[v][1] + self.norm[v][0]
                for i, v in enumerate(self.variables)}

    def predict_fields(self, start_state, t_out: int) -> dict:
        return self.decode_state(self.step_state(start_state, 0, t_out))

    def close(self):
        self._ds.close()


def build_physical_model(family: str, m: dict, n_vars: int = 4) -> torch.nn.Module:
    """FNO / CNO / ViT exactly as trained, from the `model:` block of their config."""
    if family == "fno":
        from fno_operator.src.model import ConditionedFNO2d, ConditionedFNO2dConfig
        return ConditionedFNO2d(ConditionedFNO2dConfig(
            num_input_vars=n_vars, num_output_vars=n_vars,
            width=int(m.get("width", 96)), depth=int(m.get("depth", 8)),
            modes_x=int(m.get("modes_x", 16)), modes_y=int(m.get("modes_y", 16)),
            padding=int(m.get("padding", 8)),
            max_time_index=int(m.get("max_time_index", 14)),
            use_coord_features=bool(m.get("use_coord_features", True)),
        ))
    elif family == "vit":
        from vit_operator.src.model import (ContinuousViTOperator,
                                            ContinuousViTOperatorConfig)
        return ContinuousViTOperator(ContinuousViTOperatorConfig(
            num_input_vars=n_vars, num_output_vars=n_vars,
            embed_dim=int(m.get("embed_dim", 512)), depth=int(m.get("depth", 12)),
            num_heads=int(m.get("num_heads", 8)), mlp_ratio=float(m.get("mlp_ratio", 4.0)),
            patch_size=int(m.get("patch_size", 4)),
            dropout=float(m.get("dropout", 0.0)),
            attn_dropout=float(m.get("attn_dropout", 0.0)),
            max_time_index=int(m.get("max_time_index", 14)),
            use_coord_features=bool(m.get("use_coord_features", True)),
        ))
    else:
        from cno_operator.src.model import ContinuousCNO2d, ContinuousCNO2dConfig
        return ContinuousCNO2d(ContinuousCNO2dConfig(
            num_input_vars=n_vars, num_output_vars=n_vars,
            base_width=int(m.get("base_width", 76)), num_levels=int(m.get("num_levels", 4)),
            blocks_per_level=int(m.get("blocks_per_level", 2)),
            bottleneck_blocks=int(m.get("bottleneck_blocks", 2)),
            channel_multiplier=int(m.get("channel_multiplier", 2)),
            dropout=float(m.get("dropout", 0.0)),
            max_time_index=int(m.get("max_time_index", 14)),
            use_coord_features=bool(m.get("use_coord_features", True)),
        ))


class PhysicalRunner:
    """FNO / CNO / ViT physical-space baselines (single direct hop 0 -> t_out).

    All three share the exact same I/O contract: forward(state_norm[B,C,H,W],
    lead=t_out-t_in) -> normalized fields [B,C,H,W]. ViT's own test.py uses
    lead = t_out - t_in and feeds the normalized prediction straight back, so
    the step/decode/predict path below is identical across the three.
    """

    def __init__(self, entry: dict, ds_cfg: dict, decoders: dict, device):
        self.device = device
        self.variables = list(ds_cfg["variables"])
        self.norm = {k: (v["mean"], v["std"]) for k, v in ds_cfg["normalization"].items()}
        op_cfg = OmegaConf.to_container(OmegaConf.load(str(REPO / entry["op_config"])), resolve=True)
        m = op_cfg["model"]
        if entry["family"] == "vit":
            # ViT's train.py normalizes and denormalizes using the stats in its
            # OWN config; for RC/RKH those are KH's stats (a copy-paste bug in
            # the vit configs). The model therefore learned in KH-normalized
            # space even on RC/RKH, so we MUST evaluate with the same stats --
            # feeding registry (correct) stats mis-scales the input and denorm
            # and makes a working model look collapsed. FNO/CNO configs match
            # the registry, so this override is a no-op for them by design.
            vnorm = op_cfg.get("dataset", {}).get("normalization") or op_cfg.get("normalization")
            if vnorm:
                self.norm = {k: (float(v["mean"]), float(v["std"])) for k, v in vnorm.items()}
        self.model = build_physical_model(entry["family"], m, len(self.variables)).to(device)
        state, payload = _load_ckpt_state(entry["checkpoint"], device)
        res = self.model.load_state_dict(state, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"{entry['checkpoint']}: {len(res.missing_keys)} missing keys")
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.trained_epoch = _payload_int(payload, "epoch")
        self.trained_step = _payload_int(payload, "step")
        self._src = nc.Dataset(ds_cfg["source_fields"], "r")
        self._stats = {v: (float(self.norm[v][0]), float(self.norm[v][1])) for v in self.variables}

    def make_start_state(self, member_index: int):
        raw = read_fields(self._src, member_index=member_index, time_idx=0,
                          variables=self.variables)
        return torch.from_numpy(
            normalize_fields(raw, self.variables, self._stats)).float()

    @torch.no_grad()
    def step_state(self, state, t_in: int, t_out: int):
        lead = torch.tensor([int(t_out) - int(t_in)], dtype=torch.long, device=self.device)
        pred = self.model(state.unsqueeze(0).to(self.device), lead)[0]
        return pred.float().cpu()          # normalized fields, feed back directly

    def decode_state(self, state) -> dict:
        arr = state.numpy()
        return {v: arr[i] * self.norm[v][1] + self.norm[v][0]
                for i, v in enumerate(self.variables)}

    def predict_fields(self, start_state, t_out: int) -> dict:
        return self.decode_state(self.step_state(start_state, 0, t_out))

    def close(self):
        self._src.close()


class ContinuousLatentRunner:
    """Continuous-latent transformer: predict latents, decode via the frozen AE.

    State carried through a rollout is the LATENT tensor [V, C, H, W], so
    autoregressive chaining needs no quantisation step at all -- the model's
    output is directly the next input (the cleanest possible rollout).
    """

    def __init__(self, entry: dict, ds_cfg: dict, decoders: dict, device):
        from omegaconf import OmegaConf as _OC

        from tokens.encode_continuous_latents import load_continuous_ae
        from baselines.models_continuous import build_continuous_operator

        self.device = device
        self.variables = list(ds_cfg["variables"])
        self.norm = {k: (v["mean"], v["std"]) for k, v in ds_cfg["normalization"].items()}
        train_cfg = _OC.to_container(_OC.load(REPO / entry["train_config"]), resolve=True)
        self.model = build_continuous_operator(train_cfg).to(device)
        state, payload = _load_ckpt_state(entry["checkpoint"], device)
        res = self.model.load_state_dict(state, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"{entry['checkpoint']}: {len(res.missing_keys)} missing keys")
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.trained_epoch = _payload_int(payload, "epoch") if _payload_int(payload, "epoch") is not None else -1
        self.trained_step = _payload_int(payload, "step")

        self.ae = load_continuous_ae(device)
        self.latent_path = entry["latent_dataset"]
        self._ds = nc.Dataset(self.latent_path, "r")
        self._ds.set_auto_mask(False)
        self.time_indices = [int(t) for t in np.asarray(self._ds.getncattr("time_indices"))]
        self._slot = {t: i for i, t in enumerate(self.time_indices)}

    def make_start_state(self, member_index: int):
        z = np.asarray(self._ds.variables["latents"][member_index, self._slot[0]],
                       dtype=np.float32)
        return torch.from_numpy(z)                      # [V, C, H, W]

    @torch.no_grad()
    def step_state(self, state, t_in: int, t_out: int):
        out = self.model(state.unsqueeze(0).to(self.device),
                         torch.tensor([int(t_in)], dtype=torch.long, device=self.device),
                         torch.tensor([int(t_out)], dtype=torch.long, device=self.device))
        return out[0].float().cpu()

    @torch.no_grad()
    def decode_state(self, state) -> dict:
        # AE decodes one variable at a time: [1, C, H, W] -> [1, 1, 128, 128]
        z = state.to(self.device)
        recon = []
        for vi in range(z.shape[0]):
            recon.append(self.ae(z[vi:vi + 1], mode="decode")[0, 0].float().cpu().numpy())
        return {v: recon[i] * self.norm[v][1] + self.norm[v][0]
                for i, v in enumerate(self.variables)}

    def predict_fields(self, start_state, t_out: int) -> dict:
        return self.decode_state(self.step_state(start_state, 0, t_out))

    def close(self):
        self._ds.close()


class ArchDualSequentialVQ2Runner:
    """VQ-VAE2-token model: dual_sequential FA2 backbone + VQ-VAE2 decoder.

    Token layout (see VQVAE2_tokens/README.txt): the "amp" stream holds TOP
    tokens (vocab 4096) whose native grid is 16x16, stored 2x2-replicated onto
    32x32; the "morph" stream holds BOTTOM tokens (vocab 16384) natively at
    32x32. The model predicts all 32x32 amp positions independently, so a
    prediction need NOT be 2x2-constant -- we collapse each 2x2 block back to
    one top token by majority vote (exact for ground truth, and uses all four
    predictions rather than discarding three).
    """

    AMP_VOCAB, MORPH_VOCAB = 4096, 16384

    def __init__(self, entry: dict, ds_cfg: dict, decoders: dict, device):
        from baselines.arch.models import BUILDERS

        from tokens.encode_vqvae2_tokens import decode_tokens_pair, load_vqvae2

        self.device = device
        self.variables = list(ds_cfg["variables"])
        self.norm = {k: (v["mean"], v["std"]) for k, v in ds_cfg["normalization"].items()}
        state, payload = _load_ckpt_state(entry["checkpoint"], device)
        self.model = BUILDERS["dual_sequential"](_arch_config(entry, payload)).to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.trained_epoch = _payload_int(payload, "epoch") if _payload_int(payload, "epoch") is not None else -1
        self.trained_step = _payload_int(payload, "step")

        self.ae = load_vqvae2(device)
        self._decode_pair = decode_tokens_pair
        self.token_path = ds_cfg["vqvae2_tokens"]
        self._ds = nc.Dataset(self.token_path, "r")
        self.morph_offset = (int(self._ds.getncattr("morphology_offset"))
                             if "morphology_offset" in self._ds.ncattrs() else 0)

    def make_start_state(self, member_index: int):
        return _load_member_tokens_phaedra(
            self._ds, member_index=member_index, time_idx=0,
            variables=self.variables, morph_offset=self.morph_offset)

    @torch.no_grad()
    def step_state(self, state, t_in: int, t_out: int):
        amp0, morph0 = state
        pred_amp, pred_morph = self.model.predict(
            input_amp=amp0.unsqueeze(0).to(self.device),
            input_morph=morph0.unsqueeze(0).to(self.device),
            input_time_idx=torch.tensor([int(t_in)], dtype=torch.long, device=self.device),
            output_time_idx=torch.tensor([int(t_out)], dtype=torch.long, device=self.device),
            input_var_ids=None, output_var_ids=None,
            input_var_mask=None, output_var_mask=None,
            problem_type_id=torch.zeros((1,), dtype=torch.long, device=self.device),
        )
        V, H, W = amp0.shape
        pa = pred_amp.view(1, V, H, W)[0].cpu()
        pm = pred_morph.view(1, V, H, W)[0].cpu()
        return _clip_tokens_for_decoder_phaedra(pa, pm, self.AMP_VOCAB, self.MORPH_VOCAB)

    @staticmethod
    def _collapse_top(amp32: np.ndarray) -> np.ndarray:
        """[V,32,32] -> [V,16,16] by per-2x2-block majority vote."""
        c = np.stack([amp32[:, ::2, ::2], amp32[:, 1::2, ::2],
                      amp32[:, ::2, 1::2], amp32[:, 1::2, 1::2]], axis=0)  # [4,V,16,16]
        counts = np.stack([(c == c[i]).sum(axis=0) for i in range(4)], axis=0)  # [4,V,16,16]
        pick = counts.argmax(axis=0)                                            # [V,16,16]
        return np.take_along_axis(c, pick[None], axis=0)[0]

    @torch.no_grad()
    def decode_state(self, state) -> dict:
        pa, pm = state
        amp32 = pa.numpy() if torch.is_tensor(pa) else np.asarray(pa)
        bot = pm.numpy() if torch.is_tensor(pm) else np.asarray(pm)
        top = self._collapse_top(amp32)
        recon = []
        for vi in range(len(self.variables)):
            recon.append(self._decode_pair(self.ae, top[vi][None], bot[vi][None], self.device)[0])
        return {v: recon[i] * self.norm[v][1] + self.norm[v][0]
                for i, v in enumerate(self.variables)}

    def predict_fields(self, start_state, t_out: int) -> dict:
        return self.decode_state(self.step_state(start_state, 0, t_out))

    def close(self):
        self._ds.close()


FAMILIES = {
    "hub_phaedra": HubTokenRunner,
    "hub_fsq": HubTokenRunner,
    "arch_dual_sequential": ArchDualSequentialRunner,
    "fno": PhysicalRunner,
    "cno": PhysicalRunner,
    "vit": PhysicalRunner,
    "continuous": ContinuousLatentRunner,
    "arch_dual_sequential_vq2": ArchDualSequentialVQ2Runner,
}


def _test_members(token_or_source_path: str) -> tuple[list[int], list[int]]:
    ds = nc.Dataset(token_or_source_path, "r")
    vals = (np.asarray(ds.variables["member"][:], dtype=np.int64)
            if "member" in ds.variables
            else np.arange(int(ds.dimensions["member"].size), dtype=np.int64))
    n_total = int(ds.dimensions["member"].size)
    s, e = (str(ds.getncattr("split_test_range")).split(":")
            if "split_test_range" in ds.ncattrs() else (str(n_total - 240), str(n_total)))
    idx = [int(i) for i, m in enumerate(vals) if int(s) <= int(m) < int(e)]
    ids = [int(vals[i]) for i in idx]
    ds.close()
    return idx, ids


def _replace_rows(csv_path: Path, new_rows: list[dict], model: str, dataset: str) -> None:
    """Atomic-ish: drop existing rows for (model, dataset), append new ones."""
    old: list[dict] = []
    if csv_path.exists():
        with csv_path.open() as fh:
            old = [r for r in csv.DictReader(fh)
                   if not (r["model"] == model and r["dataset"] == dataset)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(old)
        w.writerows(new_rows)
    tmp.replace(csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-sample per-timestep relL1 evaluator")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-members", type=int, default=None)
    ap.add_argument("--timesteps", nargs="*", type=int, default=[2, 4, 6, 8, 10, 12, 14])
    ap.add_argument("--mode", choices=["direct", "rollout2", "rollout662", "rollout_g6"],
                    default="direct",
                    help="direct: single hop 0->t for each t. rollout2: autoregressive "
                         "step-2 chain 0->2->...->14, error recorded at every visited t. "
                         "rollout662: coarse chain 0->6->12->14 (hops 6,6,2), the "
                         "Poseidon-style strategy that minimises last-step error; records "
                         "t=6,12,14. rollout_g6: greedy coarsest-first per target (as many "
                         "6-hops as fit + remainder), i.e. 0->2,0->4,0->6,0->6->8,... . "
                         "Each non-direct mode writes its own CSV.")
    ap.add_argument("--registry", type=str, default=str(REGISTRY),
                    help="Registry YAML. Point at a snapshot registry to evaluate "
                         "frozen copies of checkpoints that live training is still "
                         "overwriting.")
    ap.add_argument("--shard-tag", type=str, default=None,
                    help="Write to per_sample_relL1.<TAG>.csv and a matching meta "
                         "file instead of the shared ones. REQUIRED when several "
                         "eval jobs run concurrently: the writer is read-modify-write "
                         "through a fixed .tmp name, so concurrent jobs would drop "
                         "each other's rows. Merge with evaluation.merge_eval_shards.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Directory for the per-sample CSV + meta JSON (default: "
                         "$PHAEDRA_OUTPUT_ROOT/evaluation/tables).")
    ap.add_argument("--deterministic", action="store_true",
                    help="Bit-reproducible eval: disable TF32, force deterministic "
                         "algorithms, seed everything. Removes the run-to-run variance "
                         "in the argmax token rollout (a TF32 logit perturbation can "
                         "flip a near-tie token and compound over hops). Needs "
                         "CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment.")
    args = ap.parse_args()

    out_csv_base, meta_json = OUT_CSV, META_JSON
    if args.out_dir:
        _d = Path(args.out_dir)
        _d.mkdir(parents=True, exist_ok=True)
        out_csv_base, meta_json = _d / OUT_CSV.name, _d / META_JSON.name
    if args.shard_tag:
        out_csv_base = out_csv_base.with_name(f"per_sample_relL1.{args.shard_tag}.csv")
        meta_json = meta_json.with_name(
            f"per_sample_relL1.meta.{args.shard_tag}.json")

    reg = OmegaConf.to_container(OmegaConf.load(args.registry), resolve=True)
    names = list(reg["models"].keys()) if args.all else (args.models or [])
    if not names:
        raise SystemExit("Pass --all or --models <name...>")

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        configure_torch(tf32=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        print("[eval] DETERMINISTic mode: TF32 off, deterministic algorithms on, "
              f"seed=0, CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
              flush=True)
    else:
        configure_torch(tf32=True)
    device = get_device()
    git_commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()

    meta = json.loads(meta_json.read_text()) if meta_json.exists() else {}
    for name in names:
        entry = reg["models"][name]
        ds_cfg = reg["datasets"][entry["dataset"]]
        t0 = time.time()
        runner = FAMILIES[entry["family"]](entry, ds_cfg, reg["decoders"], device)
        print(f"[eval] {name}: family={entry['family']} params={runner.n_params:,}", flush=True)

        # Member list from the token file for token models (aligned member ids),
        # from source fields for physical models.
        # Physical-space models read the source field file, which carries no split
        # attributes; take the test split from the dataset's Phaedra token file so
        # every family is scored on exactly the same trajectories.
        member_src = (getattr(runner, "token_path", None)
                      or getattr(runner, "latent_path", None))
        if member_src is None:   # physical-space model: split from the Phaedra token file if present,
            pt = ds_cfg.get("phaedra_tokens")   # else the last 240 members of the field file
            member_src = pt if pt and Path(pt).exists() else ds_cfg["source_fields"]
        m_idx, m_ids = _test_members(member_src)
        if args.max_members:
            m_idx, m_ids = m_idx[: args.max_members], m_ids[: args.max_members]

        src = nc.Dataset(ds_cfg["source_fields"], "r")
        src_vals = (np.asarray(src.variables["member"][:], dtype=np.int64)
                    if "member" in src.variables else None)
        src_lookup = ({int(m): int(i) for i, m in enumerate(src_vals)}
                      if src_vals is not None else None)

        rows: list[dict] = []
        for k, (mi, mid) in enumerate(zip(m_idx, m_ids)):
            if k % 40 == 0:
                print(f"[eval] {name}: member {k + 1}/{len(m_idx)}", flush=True)
            start = runner.make_start_state(mi)
            si = src_lookup.get(mid, mid) if src_lookup else mid

            preds: dict[int, dict] = {}
            if args.mode == "direct":
                for t_out in args.timesteps:
                    preds[t_out] = runner.predict_fields(start, t_out)
            elif args.mode == "rollout_g6":
                # Greedy coarsest-first schedule per target: as many 6-hops as
                # fit, then one final hop of the remainder. Each target gets its
                # own chain (0->6->8 for t=8, 0->6->12->14 for t=14, 0->2 for t=2).
                for t_out in args.timesteps:
                    hops = [6] * (t_out // 6) + ([t_out % 6] if t_out % 6 else [])
                    state, t_cur = start, 0
                    for hop in hops:
                        state = runner.step_state(state, t_cur, t_cur + hop)
                        t_cur += hop
                    preds[t_out] = runner.decode_state(state)
            else:
                # Autoregressive chain with a fixed hop schedule to t=14; record
                # every visited step that is requested. rollout2 = seven 2-hops;
                # rollout662 = 0->6->12->14 (hops 6,6,2).
                schedule = (6, 6, 2) if args.mode == "rollout662" else (2,) * 7
                state, t_cur = start, 0
                for hop in schedule:
                    state = runner.step_state(state, t_cur, t_cur + hop)
                    t_cur += hop
                    if t_cur in args.timesteps:
                        preds[t_cur] = runner.decode_state(state)

            for t_out, pred in preds.items():
                true_raw = read_fields(src, member_index=si, time_idx=int(t_out),
                                       variables=runner.variables)
                for vi, var in enumerate(runner.variables):
                    rows.append({
                        "model": name, "dataset": entry["dataset"], "traj_id": mid,
                        "var": var, "t": int(t_out),
                        "relL1": f"{_relative_l1(pred[var], true_raw[vi]):.8f}",
                    })
        src.close()
        runner.close()

        # direct -> per_sample_relL1.csv; rollout2 -> ..._rollout.csv;
        # rollout662 -> ..._rollout662.csv (each mode isolated).
        _suffix = {"rollout2": "rollout", "rollout662": "rollout662",
                   "rollout_g6": "rollout_g6"}
        out_csv = (out_csv_base if args.mode == "direct"
                   else out_csv_base.with_name(
                       out_csv_base.name.replace(
                           "per_sample_relL1",
                           f"per_sample_relL1_{_suffix[args.mode]}", 1)))
        _replace_rows(out_csv, rows, name, entry["dataset"])
        wall = time.time() - t0
        meta_key = name if args.mode == "direct" else f"{name}@{args.mode}"
        meta[meta_key] = {
            "family": entry["family"], "dataset": entry["dataset"], "mode": args.mode,
            "checkpoint": entry["checkpoint"], "params": int(runner.n_params),
            "trained_epoch": getattr(runner, "trained_epoch", None),
            "trained_step": getattr(runner, "trained_step", None),
            "n_members": len(m_idx), "timesteps": args.timesteps,
            "wallclock_s": round(wall, 1), "git_commit": git_commit,
            "device": str(device), "deterministic": bool(args.deterministic),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "repo": str(REPO),
        }
        meta_json.write_text(json.dumps(meta, indent=2))
        print(f"[eval] {name} [{args.mode}]: {len(rows)} rows in {wall:.0f}s -> {out_csv}",
              flush=True)


if __name__ == "__main__":
    main()
