"""Dump t=14 predictions of every registered 38M model for a few test members,
under all three strategies (direct, 2-step, 6-6-2), plus the ground truth and
the decode of the GT representation (floor). The figure builder
(verify_report.py) picks the strategy per model from the metrics table, so
the plotted field is exactly the one the table scores.

Writes results/verify/fields/fields_<ds>.npz with keys
  gt|t0|<member>, gt|t14|<member>, pred|<model>|<strategy>|<member>,
  recon|<model>|<member>   (token/latent models only)
and a JSON sidecar with per-member relL1 for each (model, strategy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf

from fno_operator.src.dataset import read_fields
from hub.trainers.test import _load_member_tokens_fsq, _load_member_tokens_phaedra, _relative_l1
from hub.utils.runtime import configure_torch, get_device
from evaluation.eval_downstream import EVAL_ROOT, FAMILIES, REPO

OUT = EVAL_ROOT / "fields"
STEMS = ["fno_38m", "cno_38m", "vit_38m", "continuous_38m", "vqvae2_38m", "fsq_38m", "phaedra_38m"]
SCHED = {"direct": (14,), "rollout2": (2,) * 7, "rollout662": (6, 6, 2)}


def _gt_recon(runner, member: int, t: int):
    name = type(runner).__name__
    if name == "PhysicalRunner":
        return None
    if name == "ContinuousLatentRunner":
        z = np.asarray(runner._ds.variables["latents"][member, runner._slot[t]], dtype=np.float32)
        return runner.decode_state(torch.from_numpy(z))
    if getattr(runner, "family", "") == "hub_fsq":
        return runner.decode_state(_load_member_tokens_fsq(runner._ds, member, t, runner.variables))
    a, m = _load_member_tokens_phaedra(runner._ds, member, t, runner.variables, runner.morph_offset)
    return runner.decode_state((a, m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(REPO / "evaluation" / "eval_registry.yaml"))
    ap.add_argument("--datasets", nargs="+", default=["kh", "rc", "rkh"])
    ap.add_argument("--members", nargs="+", type=int, default=[9760, 9761, 9762])
    ap.add_argument("--extra-models", nargs="*", default=[])
    args = ap.parse_args()

    configure_torch(tf32=False)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0)
    device = get_device()
    reg = OmegaConf.to_container(OmegaConf.load(args.registry), resolve=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        ds_cfg = reg["datasets"][ds]
        variables = list(ds_cfg["variables"])
        src = nc.Dataset(ds_cfg["source_fields"], "r")
        arrays: dict[str, np.ndarray] = {}
        errs: dict = {}
        for mid in args.members:
            arrays[f"gt|t0|{mid}"] = read_fields(src, member_index=mid, time_idx=0, variables=variables)
            arrays[f"gt|t14|{mid}"] = read_fields(src, member_index=mid, time_idx=14, variables=variables)
        models = [f"{s}_{ds}" for s in STEMS] + [m for m in args.extra_models
                                                 if reg["models"].get(m, {}).get("dataset") == ds]
        for name in models:
            if name not in reg["models"]:
                print(f"[dump] {name}: not in registry, skipped", flush=True); continue
            entry = reg["models"][name]
            runner = FAMILIES[entry["family"]](entry, ds_cfg, reg["decoders"], device)
            print(f"[dump] {ds} {name} ({entry['family']})", flush=True)
            for mid in args.members:
                start = runner.make_start_state(mid)
                for strat, hops in SCHED.items():
                    state, t = start, 0
                    for h in hops:
                        state = runner.step_state(state, t, t + h); t += h
                    pred = runner.decode_state(state)
                    arr = np.stack([pred[v] for v in variables]).astype(np.float32)
                    arrays[f"pred|{name}|{strat}|{mid}"] = arr
                    errs[f"{name}|{strat}|{mid}"] = {v: _relative_l1(arr[i], arrays[f"gt|t14|{mid}"][i])
                                                     for i, v in enumerate(variables)}
                rec = _gt_recon(runner, mid, 14)
                if rec is not None:
                    arr = np.stack([rec[v] for v in variables]).astype(np.float32)
                    arrays[f"recon|{name}|{mid}"] = arr
                    errs[f"{name}|recon|{mid}"] = {v: _relative_l1(arr[i], arrays[f"gt|t14|{mid}"][i])
                                                   for i, v in enumerate(variables)}
            runner.close()
            del runner; torch.cuda.empty_cache()
        src.close()
        np.savez_compressed(OUT / f"fields_{ds}.npz", **arrays)
        (OUT / f"fields_{ds}.json").write_text(json.dumps({"variables": variables, "members": args.members,
                                                            "models": models, "relL1": errs}, indent=1))
        print(f"[dump] wrote {OUT / f'fields_{ds}.npz'} ({len(arrays)} arrays)", flush=True)


if __name__ == "__main__":
    main()
