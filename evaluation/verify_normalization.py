"""Normalization / decode-floor audit for every token- or latent-based model.

For each dataset (KH / RC / RKH) and each representation (Phaedra, FSQ,
VQ-VAE-2, continuous AE) this script:

  1. decodes the GROUND-TRUTH representation of N test trajectories at
     t in {0, 14}, denormalizes with the registry (mean, std), and reports the
     relative L1 vs the source field. This is the reconstruction floor; it can
     only be small if the (mean, std) used at decode time match the ones used
     when the tokens/latents were generated. A wrong std shows up as a
     large error, a wrong mean as a bias.
  2. repeats the denormalization with the stats the ORIGINAL paper's hub
     test.py resolved from masked_discrete_diffusion_pde/config_kh.yaml
     (RC v-mean 0.00042 there vs 0.0026327 in the registry / token-gen config)
     to quantify what that slip changes, and with a deliberately wrong set
     (KH stats applied to RC/RKH) to show the check's sensitivity.
  3. recomputes the per-variable mean/std of the source data over a member
     subset and compares to the registry, so the registry values themselves
     are audited, not just their consistency.

Writes results/verify/tables/normalization_check.{md,csv,json}.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf

from fno_operator.src.dataset import read_fields
from hub.trainers.test import (_decode_normalized_fields, _load_member_tokens_fsq,
                               _load_member_tokens_phaedra, _relative_l1)
from hub.utils.phaedra_decoder import load_token_decoder
from hub.utils.runtime import configure_torch, get_device
from evaluation.eval_downstream import EVAL_ROOT, REPO

PAPER_DENORM_CFG = REPO / "evaluation" / "denormalization.yaml"
PAPER_NAME = {"kh": "CEU_2D_KelvinHelmholtzLowRes", "rc": "CEU_2D_RiemannCurvedLowRes",
              "rkh": "CEU_2D_RiemannKelvinHelmholtzLowRes"}
OUT = EVAL_ROOT / "tables"


def _paper_stats(ds: str, variables: list[str]) -> dict[str, tuple[float, float]]:
    cfg = OmegaConf.to_container(OmegaConf.load(PAPER_DENORM_CFG), resolve=True)
    for e in cfg["denormalization"]["datasets"]:
        if e["name"] == PAPER_NAME[ds]:
            return {v: (float(m), float(s)) for v, m, s in
                    zip(e["field_variables_out"], e["_normalization_mean"], e["_normalization_std"])}
    raise KeyError(ds)


def _source_stats(src: nc.Dataset, variables: list[str], members: list[int]) -> dict:
    out = {}
    for var in variables:
        s = ss = 0.0; n = 0
        for mi in members:
            a = np.asarray(src.variables[var][mi], dtype=np.float64)   # [21,128,128]
            s += a.sum(); ss += (a * a).sum(); n += a.size
        mean = s / n
        out[var] = (float(mean), float(np.sqrt(max(ss / n - mean * mean, 0.0))))
    return out


def _rel(pred: np.ndarray, true: np.ndarray) -> float:
    return _relative_l1(pred, true)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(REPO / "evaluation" / "eval_registry.yaml"))
    ap.add_argument("--members", type=int, default=16, help="test members per check")
    ap.add_argument("--stat-members", type=int, default=300,
                    help="train members for the source-stat recomputation")
    ap.add_argument("--datasets", nargs="+", default=["kh", "rc", "rkh"])
    args = ap.parse_args()

    configure_torch(tf32=False)
    device = get_device()
    reg = OmegaConf.to_container(OmegaConf.load(args.registry), resolve=True)
    OUT.mkdir(parents=True, exist_ok=True)

    from tokens.encode_continuous_latents import load_continuous_ae
    from tokens.encode_vqvae2_tokens import decode_tokens_pair, load_vqvae2

    dec_ph = load_token_decoder({**reg["decoders"]["phaedra"], "enabled": True}, device=device)
    dec_fsq = load_token_decoder({**reg["decoders"]["fsq"], "enabled": True}, device=device)
    vq2 = load_vqvae2(device)
    cae = load_continuous_ae(device)

    rows: list[dict] = []
    stat_rows: list[dict] = []
    for ds in args.datasets:
        cfg = reg["datasets"][ds]
        variables = list(cfg["variables"])
        reg_stats = {k: (float(v["mean"]), float(v["std"])) for k, v in cfg["normalization"].items()}
        pap_stats = _paper_stats(ds, variables)
        kh_stats = {k: (float(v["mean"]), float(v["std"]))
                    for k, v in reg["datasets"]["kh"]["normalization"].items()}
        src = nc.Dataset(cfg["source_fields"], "r")

        # ---- 3. audit the registry stats against the data ----
        n_mem = int(src.dimensions["member"].size)
        rng = np.random.default_rng(0)
        # training split from the token file's attribute (default: all but the last 360)
        with nc.Dataset(cfg["phaedra_tokens"], "r") as _tok:
            _rng = str(_tok.getncattr("split_train_range")) if "split_train_range" in _tok.ncattrs() else f"0:{max(1, n_mem - 360)}"
        n_train = int(_rng.split(":")[1])
        train_members = sorted(rng.choice(n_train, size=min(args.stat_members, n_train), replace=False).tolist())
        emp = _source_stats(src, variables, train_members)
        for var in variables:
            stat_rows.append({"dataset": ds, "var": var,
                              "registry_mean": reg_stats[var][0], "registry_std": reg_stats[var][1],
                              "paper_cfg_mean": pap_stats[var][0], "paper_cfg_std": pap_stats[var][1],
                              "empirical_mean": emp[var][0], "empirical_std": emp[var][1],
                              "n_train_members": len(train_members)})
        print(f"[norm] {ds}: registry vs empirical({len(train_members)} train members):", flush=True)
        for var in variables:
            print(f"        {var:4s} reg=({reg_stats[var][0]:+.6f},{reg_stats[var][1]:.6f}) "
                  f"paper=({pap_stats[var][0]:+.6f},{pap_stats[var][1]:.6f}) "
                  f"emp=({emp[var][0]:+.6f},{emp[var][1]:.6f})", flush=True)

        # ---- 1./2. decode floors under several denorm choices ----
        ph = nc.Dataset(cfg["phaedra_tokens"], "r")
        fq = nc.Dataset(cfg["fsq_tokens"], "r")
        vq = nc.Dataset(cfg["vqvae2_tokens"], "r")
        ct = nc.Dataset(reg["models"][f"continuous_38m_{ds}"]["latent_dataset"], "r")
        ct.set_auto_mask(False)
        ct_slot = {int(t): i for i, t in enumerate(np.asarray(ct.getncattr("time_indices")))}
        ct_mean = np.asarray(ct.getncattr("normalization_mean"), dtype=np.float64)
        ct_std = np.asarray(ct.getncattr("normalization_std"), dtype=np.float64)
        file_stats = {v: (float(ct_mean[i]), float(ct_std[i])) for i, v in enumerate(variables)}
        ph_off = int(ph.getncattr("morphology_offset")) if "morphology_offset" in ph.ncattrs() else 0
        vq_off = int(vq.getncattr("morphology_offset")) if "morphology_offset" in vq.ncattrs() else 0

        stat_sets = {"registry": reg_stats, "paper_cfg": pap_stats, "latent_file_attrs": file_stats,
                     "WRONG_kh_stats": kh_stats}
        # first `--members` trajectories of the test split (from the token file's attribute)
        with nc.Dataset(cfg["phaedra_tokens"], "r") as _tok:
            _t = str(_tok.getncattr("split_test_range")) if "split_test_range" in _tok.ncattrs() else f"{n_mem - 240}:{n_mem}"
        t0, t1 = (int(x) for x in _t.split(":"))
        members = list(range(t0, min(t1, t0 + args.members)))
        acc: dict = {}
        for t in (0, 14):
            for mi in members:
                true = read_fields(src, member_index=mi, time_idx=t, variables=variables)
                recs = {}
                with torch.no_grad():
                    a, m = _load_member_tokens_phaedra(ph, mi, t, variables, ph_off)
                    recs["phaedra"] = _decode_normalized_fields(dec_ph, "phaedra", amp_tokens=a, morph_tokens=m)
                    f = _load_member_tokens_fsq(fq, mi, t, variables)
                    recs["fsq"] = _decode_normalized_fields(dec_fsq, "fsq", fsq_tokens=f)
                    a2, m2 = _load_member_tokens_phaedra(vq, mi, t, variables, vq_off)
                    top = a2.numpy()[:, ::2, ::2]
                    recs["vqvae2"] = np.stack([decode_tokens_pair(vq2, top[vi][None], m2.numpy()[vi][None], device)[0]
                                               for vi in range(len(variables))])
                    z = torch.from_numpy(np.asarray(ct.variables["latents"][mi, ct_slot[t]], dtype=np.float32)).to(device)
                    recs["continuous"] = np.stack([cae(z[vi:vi + 1], mode="decode")[0, 0].float().cpu().numpy()
                                                   for vi in range(len(variables))])
                for rep, rec in recs.items():
                    for sname, st in stat_sets.items():
                        if sname == "latent_file_attrs" and rep != "continuous":
                            continue
                        for vi, var in enumerate(variables):
                            phys = rec[vi] * st[var][1] + st[var][0]
                            acc.setdefault((rep, sname, t, var), []).append(_rel(phys, true[vi]))
        for (rep, sname, t, var), vals in sorted(acc.items()):
            rows.append({"dataset": ds, "representation": rep, "denorm_stats": sname, "t": t,
                         "var": var, "relL1_mean": float(np.mean(vals)), "n": len(vals)})
        for rep in ("phaedra", "fsq", "vqvae2", "continuous"):
            for sname in stat_sets:
                if sname == "latent_file_attrs" and rep != "continuous":
                    continue
                avg = {t: np.mean([np.mean(acc[(rep, sname, t, v)]) for v in variables]) for t in (0, 14)}
                print(f"[floor] {ds} {rep:10s} {sname:16s} t=0 {avg[0]:.4f}  t=14 {avg[14]:.4f}", flush=True)
        for d in (ph, fq, vq, ct, src):
            d.close()

    write_outputs(rows, stat_rows, args)


def write_outputs(rows, stat_rows, args) -> None:
    with (OUT / "normalization_check.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (OUT / "normalization_stats.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys())); w.writeheader(); w.writerows(stat_rows)
    (OUT / "normalization_check.json").write_text(json.dumps({"floors": rows, "stats": stat_rows}, indent=1))

    # ---- markdown ----
    md = ["# Normalization / decode-floor audit", "",
          f"Decode floors: GT representation decoded and denormalized, relative L1 vs source, "
          f"mean over {args.members} test trajectories. `registry` = evaluation/eval_registry.yaml "
          f"stats (= the token-generation configs); `paper_cfg` = stats in evaluation/denormalization.yaml "
          f"(used by hub.trainers.test); `WRONG_kh_stats` = KH stats applied to every "
          f"dataset (sensitivity check).", "",
          "## Registry vs empirical source statistics", "",
          "| Data | var | registry mean / std | paper-cfg mean / std | empirical mean / std (train subset) |",
          "|---|---|---|---|---|"]
    for r in stat_rows:
        flag = "" if abs(r["registry_mean"] - r["paper_cfg_mean"]) < 1e-9 and abs(r["registry_std"] - r["paper_cfg_std"]) < 1e-9 else " **≠**"
        md.append(f"| {r['dataset'].upper()} | {r['var']} | {r['registry_mean']:+.6f} / {r['registry_std']:.6f} | "
                  f"{r['paper_cfg_mean']:+.6f} / {r['paper_cfg_std']:.6f}{flag} | "
                  f"{r['empirical_mean']:+.6f} / {r['empirical_std']:.6f} |")
    md += ["", "## Decode floors (relative L1, %), averaged over rho,u,v,p", "",
           "| Data | representation | denorm stats | t=0 | t=14 | per-var t=14 (rho / u / v / p) |", "|---|---|---|---|---|---|"]
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["representation"], r["denorm_stats"], r["t"]), {})[r["var"]] = r["relL1_mean"]
    for ds in args.datasets:
        for rep in ("phaedra", "fsq", "vqvae2", "continuous"):
            for sname in ("registry", "paper_cfg", "latent_file_attrs", "WRONG_kh_stats"):
                k0, k14 = (ds, rep, sname, 0), (ds, rep, sname, 14)
                if k14 not in by:
                    continue
                # KH: v == 0 identically at t=0, so its relative L1 is undefined there;
                # average the t=0 floor over rho,u,p only and flag it.
                v0 = {v: x for v, x in by[k0].items() if not (ds == "kh" and v == "v")}
                a0 = 100 * np.mean(list(v0.values())); a14 = 100 * np.mean(list(by[k14].values()))
                a0s = f"{a0:.2f}" + ("†" if ds == "kh" else "")
                pv = " / ".join(f"{100 * by[k14][v]:.2f}" for v in ("rho", "u", "v", "p"))
                md.append(f"| {ds.upper()} | {rep} | {sname} | {a0s} | {a14:.2f} | {pv} |")
    md += ["", "† KH: $v \\equiv 0$ at $t=0$ (relative $L_1$ undefined), so the KH $t=0$ column averages ρ, u, p only."]
    (OUT / "normalization_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def rebuild_markdown_from_csv() -> None:
    """Re-render the .md from the saved CSVs (no GPU needed)."""
    import argparse as _ap
    rows = list(csv.DictReader((OUT / "normalization_check.csv").open()))
    for r in rows:
        r["t"] = int(r["t"]); r["relL1_mean"] = float(r["relL1_mean"])
    stat_rows = list(csv.DictReader((OUT / "normalization_stats.csv").open()))
    for r in stat_rows:
        for k in list(r):
            if k not in ("dataset", "var"):
                r[k] = float(r[k])
    n = max(int(r["n"]) for r in rows)
    write_outputs(rows, stat_rows, _ap.Namespace(members=n, datasets=["kh", "rc", "rkh"]))


if __name__ == "__main__":
    import sys as _sys
    if "--rebuild-md" in _sys.argv:
        rebuild_markdown_from_csv()
    else:
        main()
