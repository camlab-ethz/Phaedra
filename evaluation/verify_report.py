"""Tables, figures and a report from the per-trajectory evaluation CSVs written by
`python -m evaluation.eval_downstream` (one run per model and strategy, see
scripts/slurm/evaluate.sbatch), all under $PHAEDRA_OUTPUT_ROOT/evaluation:

  tables/per_timestep_<mode>.{md,tex,csv}     avg-over-variables relL1 per timestep
  tables/final_t14_<mode>.{md,tex,csv}        per-variable + average at t=14, per strategy
  tables/final_t14_best.{md,tex,csv}          per-variable + average at t=14, best strategy
  tables/strategy_compare_t14.{md,csv}        avg at t=14 for every strategy
  figures/per_timestep_curves.{pdf,png}       error-vs-t curves (direct + 2-step), 3 panels
  figures/fields_<ds>_m<member>.{pdf,png}     GT + one row per model, 4 variables, t=14
                                              (needs the dumps of evaluation.verify_dump_fields)
  REPORT.md                                    everything, plus run metadata

All relL1 values are reported in PERCENT (0.0971 -> 9.71). "Average" = mean of
the four per-variable numbers (each itself a mean over 240 trajectories), which
is identical to the mean over trajectories of the per-trajectory variable average.
± is the 95% bootstrap CI of that mean (10,000 resamples, seeded per table cell).

Usage:  python -m evaluation.verify_report [--no-figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.eval_downstream import EVAL_ROOT

RES = EVAL_ROOT
TAB, FIG, FLD = RES / "tables", RES / "figures", RES / "fields"

VARS = ["rho", "u", "v", "p"]
VAR_TEX = {"rho": r"$\rho$", "u": "$u$", "v": "$v$", "p": "$p$"}
DATASETS = ["kh", "rc", "rkh"]
DS_PRETTY = {"kh": "KH", "rc": "RC", "rkh": "RKH"}
STEMS = ["fno_38m", "cno_38m", "vit_38m", "continuous_38m", "vqvae2_38m", "fsq_38m", "phaedra_38m"]
PRETTY = {"fno_38m": "FNO", "cno_38m": "CNO", "vit_38m": "ViT", "continuous_38m": "Continuous Transformer",
          "vqvae2_38m": "VQ-VAE-2", "fsq_38m": "FSQ", "phaedra_38m": "Phaedra"}
EXTRA: dict = {}   # extra labelled rows (model_id -> (stem, dataset, label)); none by default
# Model-id renames applied at load time (CSV model column and field-dump keys).
# The continuous RKH run diverged after epoch ~63 (val latent-L1 3.24 -> 11.3), so the
# epoch-59 checkpoint (best saved by validation loss) is the main row and the final
# checkpoint is shown as an extra, labelled row.
RENAME: dict = {}
MODES = {"direct": ("per_sample_relL1", "direct 0→t"),
         "rollout2": ("per_sample_relL1_rollout", "2-step rollout"),
         "rollout662": ("per_sample_relL1_rollout662", "6-6-2 rollout")}
TAG = {"direct": "d", "rollout2": "r2", "rollout662": "662"}
STRAT_TEX = {"direct": "direct", "rollout2": "2-step", "rollout662": "6-6-2"}
T_ALL = [2, 4, 6, 8, 10, 12, 14]
N_BOOT = 10_000


# ---------------------------------------------------------------- loading
def _model_key(model: str) -> tuple[str, str, str]:
    """model id -> (stem, dataset, label)."""
    if model in EXTRA:
        return EXTRA[model]
    for ds in DATASETS:
        if model.endswith("_" + ds):
            stem = model[: -(len(ds) + 1)]
            return stem, ds, PRETTY.get(stem, stem)
    raise ValueError(model)


def load_mode(mode: str, tab: Path = TAB, rename: bool = True) -> dict:
    """-> {(model, t): {var: np.array over trajectories (sorted by traj_id)}}"""
    stem = MODES[mode][0]
    canonical = tab / f"{stem}.csv"
    shards = sorted(p for p in tab.glob(f"{stem}.*.csv") if p != canonical)
    files = shards if shards else ([canonical] if canonical.exists() else [])
    acc: dict = defaultdict(lambda: defaultdict(dict))
    for f in files:
        with f.open() as fh:
            for r in csv.DictReader(fh):
                m = RENAME.get(r["model"], r["model"]) if rename else r["model"]
                acc[(m, int(r["t"]))][r["var"]][int(r["traj_id"])] = float(r["relL1"])
    out = {}
    for k, d in acc.items():
        if not _known(k[0]):          # e.g. scaling-sweep or masked-operator shards
            continue
        out[k] = {v: np.array([d[v][i] for i in sorted(d[v])]) for v in d}
    return out


def _known(model: str) -> bool:
    if model in EXTRA:
        return True
    return any(model == f"{s}_{ds}" for s in STEMS for ds in DATASETS)


def load_meta(tab: Path = TAB) -> dict:
    meta = {}
    for f in sorted(tab.glob("per_sample_relL1.meta.*.json")):
        for k, v in json.loads(f.read_text()).items():
            base, _, mode = k.partition("@")
            base = RENAME.get(base, base)
            meta[base + ("@" + mode if mode else "")] = v
    return meta


def ci_half(x: np.ndarray, key: str) -> float:
    """Half-width of the 95% bootstrap CI of mean(x). The resampling stream is seeded
    by the table cell (`model|strategy|t`), so a cell does not depend on which other
    models are evaluated or in which order."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    bs = x[idx].mean(axis=1)
    return float((np.percentile(bs, 97.5) - np.percentile(bs, 2.5)) / 2)


def traj_avg(d: dict) -> np.ndarray:
    """per-trajectory average over variables."""
    return np.mean(np.stack([d[v] for v in VARS]), axis=0)


# ---------------------------------------------------------------- tables
def rows_for(stats: dict, t: int, by_model: bool = True) -> list[tuple[str, str, str, str, dict]]:
    """[(model, stem, ds, label, {var: arr})] present at time t. by_model=True gives
    the paper's grouping (FNO KH/RC/RKH, CNO KH/RC/RKH, ...), else dataset-major."""
    out = []
    order = [(stem, ds) for stem in STEMS for ds in DATASETS] if by_model else \
            [(stem, ds) for ds in DATASETS for stem in STEMS]
    for stem, ds in order:
        if True:
            m = f"{stem}_{ds}"
            if (m, t) in stats:
                out.append((m, stem, ds, PRETTY[stem], stats[(m, t)]))
            for em, (es, eds, el) in EXTRA.items():
                if es == stem and eds == ds and (em, t) in stats:
                    out.append((em, stem, ds, el, stats[(em, t)]))
    return out


def per_timestep_tables(stats: dict, mode: str) -> tuple[str, str]:
    ts = [t for t in T_ALL if any(k[1] == t for k in stats)]
    md = [f"### Average relative $L_1$ (%) over $\\rho,u,v,p$ per timestep — {MODES[mode][1]}", "",
          "Mean over 240 test trajectories ± 95% bootstrap CI of the mean. **Bold** = best per dataset and timestep.", "",
          "| Model | Data | " + " | ".join(f"t={t}" for t in ts) + " |", "|---|---|" + "---|" * len(ts)]
    tex = [r"\begin{tabular}{ll" + "c" * len(ts) + "}", r"\toprule",
           "Model & Dataset & " + " & ".join(f"$t={t}$" for t in ts) + r" \\", r"\midrule"]
    csv_rows = []
    models = sorted({k[0] for k in stats}, key=lambda m: (DATASETS.index(_model_key(m)[1]),
                                                        STEMS.index(_model_key(m)[0]), m in EXTRA))
    cell = {}
    for m in models:
        for t in ts:
            if (m, t) in stats:
                a = traj_avg(stats[(m, t)])
                cell[(m, t)] = (100 * a.mean(), 100 * ci_half(a, f"{m}|{mode}|{t}"))
    best = {}
    for t in ts:
        for ds in DATASETS:
            c = [(cell[(m, t)][0], m) for m in models if (m, t) in cell and _model_key(m)[1] == ds and m not in EXTRA]
            if c:
                best[(ds, t)] = min(c)[1]
    last_ds = None
    for m in models:
        stem, ds, label = _model_key(m)
        if last_ds is not None and ds != last_ds:
            tex.append(r"\midrule")
        last_ds = ds
        mdc, txc, row = [], [], {"model": label, "model_id": m, "dataset": DS_PRETTY[ds]}
        for t in ts:
            if (m, t) not in cell:
                mdc.append("--"); txc.append("--"); row[f"t{t}"] = ""; continue
            mu, hw = cell[(m, t)]
            s, st = f"{mu:.2f} ± {hw:.2f}", f"{mu:.2f}"
            b = best.get((ds, t)) == m
            mdc.append(f"**{s}**" if b else s); txc.append(rf"\textbf{{{st}}}" if b else st)
            row[f"t{t}"] = f"{mu:.4f}"
        md.append(f"| {label} | {DS_PRETTY[ds]} | " + " | ".join(mdc) + " |")
        tex.append(f"{label} & {DS_PRETTY[ds]} & " + " & ".join(txc) + r" \\")
        csv_rows.append(row)
    tex += [r"\bottomrule", r"\end{tabular}"]
    _write_csv(TAB / f"per_timestep_{mode}.csv", csv_rows)
    (TAB / f"per_timestep_{mode}.md").write_text("\n".join(md) + "\n")
    (TAB / f"per_timestep_{mode}.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md), "\n".join(tex)


def final_table(entries: list[tuple], title: str, fname: str, note: str = "",
                caption: str = "Full per-variable operator learning relative $L_1$ errors at final time step ($t=0.7$).") -> str:
    """entries: [(model, stem, ds, label, {var: arr}, strategy)] -> md/tex/csv."""
    md = [f"### {title}", ""] + ([note, ""] if note else []) + [
        "| Model | Data | ρ | u | v | p | Average | strategy |", "|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[h]", r"    \centering", rf"    \caption{{{caption}}}",
           r"    \begin{tabular}{llccccccc}", r"        \toprule",
           r"        Model & Dataset  & $\rho$ & $u$ & $v$ & $p$ & Average & Strategy & CI$_{95}$ \\",
           r"        \midrule"]
    csv_rows, last_stem = [], None
    # bold best average per dataset (extras excluded)
    best = {}
    for m, stem, ds, label, d, strat in entries:
        if m in EXTRA:
            continue
        avg = 100 * np.mean([d[v].mean() for v in VARS])
        if ds not in best or avg < best[ds][0]:
            best[ds] = (avg, m)
    for m, stem, ds, label, d, strat in entries:
        if last_stem is not None and stem != last_stem:
            tex.append(r"        \midrule")
        last_stem = stem
        pv = [100 * d[v].mean() for v in VARS]
        avg = float(np.mean(pv))
        a = traj_avg(d)
        hw = 100 * ci_half(a, f"{m}|{strat}|14")
        b = best.get(ds, (None, None))[1] == m
        cells = [f"{x:.2f}" for x in pv] + [f"{avg:.2f} ± {hw:.2f}"]
        md.append(f"| {label} | {DS_PRETTY[ds]} | " + " | ".join(f"**{c}**" if b else c for c in cells) + f" | {TAG[strat]} |")
        texc = ([f"{x:.2f}" for x in pv] + [rf"\textbf{{{avg:.2f}}}" if b else f"{avg:.2f}"]
                + [STRAT_TEX[strat], rf"$\pm{hw:.2f}$"])
        # extra (disclosure) rows are emitted commented-out so they can be toggled in the .tex
        prefix = "        % " if m in EXTRA else "        "
        tex.append(f"{prefix}{label} & {DS_PRETTY[ds]} & " + " & ".join(texc) + r" \\")
        csv_rows.append({"model": label, "model_id": m, "dataset": DS_PRETTY[ds], **{v: f"{x:.4f}" for v, x in zip(VARS, pv)},
                         "average": f"{avg:.4f}", "ci95_half": f"{hw:.4f}", "strategy": strat, "n_traj": len(a)})
    tex += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    _write_csv(TAB / f"{fname}.csv", csv_rows)
    (TAB / f"{fname}.md").write_text("\n".join(md) + "\n")
    (TAB / f"{fname}.tex").write_text("\n".join(tex) + "\n")
    return "\n".join(md)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- figures
def fig_per_timestep(all_stats: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = [m for m in ("direct", "rollout2") if all_stats.get(m)]
    if not modes:
        return
    fig, axes = plt.subplots(len(modes), 3, figsize=(13, 3.6 * len(modes)), squeeze=False)
    colors = dict(zip(STEMS, plt.cm.tab10(np.arange(len(STEMS)))))
    for i, mode in enumerate(modes):
        st = all_stats[mode]
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            for stem in STEMS:
                m = f"{stem}_{ds}"
                pts = [(t, 100 * traj_avg(st[(m, t)]).mean()) for t in T_ALL if (m, t) in st]
                if pts:
                    ax.plot(*zip(*pts), marker="o", ms=3.5, lw=1.5, color=colors[stem],
                            label=PRETTY[stem], ls="-" if stem == "phaedra_38m" else "--" if stem in ("fsq_38m", "vqvae2_38m", "continuous_38m") else ":")
            ax.set_title(f"{DS_PRETTY[ds]} — {MODES[mode][1]}")
            ax.set_xlabel("timestep $t$ (t=14 ↔ 0.7)"); ax.set_ylabel("avg. relative $L_1$ (%)")
            ax.set_xticks(T_ALL); ax.grid(alpha=0.3)
            if ds == "rc" and mode == "direct":
                ax.set_yscale("log")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"per_timestep_curves.{ext}", dpi=160)
    plt.close(fig)


def fig_fields(ds: str, member: int, best_strat: dict) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    npz_path = FLD / f"fields_{ds}.npz"
    if not npz_path.exists():
        return None
    z0 = np.load(npz_path)
    info = json.loads((FLD / f"fields_{ds}.json").read_text())

    def _ren(key: str) -> str:
        parts = key.split("|")
        if parts[0] in ("pred", "recon"):
            parts[1] = RENAME.get(parts[1], parts[1])
        elif len(parts) == 3 and parts[1] in ("direct", "rollout2", "rollout662", "recon"):
            parts[0] = RENAME.get(parts[0], parts[0])
        return "|".join(parts)
    z = {_ren(k): z0[k] for k in z0.files}
    info["relL1"] = {_ren(k): v for k, v in info["relL1"].items()}
    if f"gt|t14|{member}" not in z:
        return None
    gt = z[f"gt|t14|{member}"]
    rows = [("Ground truth", gt)]
    for stem in STEMS:
        m = f"{stem}_{ds}"
        strat = best_strat.get(m, "direct")
        key = f"pred|{m}|{strat}|{member}"
        if key not in z:
            continue
        e = info["relL1"].get(f"{m}|{strat}|{member}", {})
        avg = 100 * np.mean([e[v] for v in VARS]) if e else float("nan")
        rows.append((f"{PRETTY[stem]}\n({TAG[strat]}, {avg:.1f}%)", z[key]))
    n = len(rows)
    fig = plt.figure(figsize=(8.6, 2.05 * n + 0.6), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 4, height_ratios=[1] * n + [0.06])
    for j, v in enumerate(VARS):
        vmin, vmax = float(gt[j].min()), float(gt[j].max())
        for i, (label, arr) in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(arr[j].T, cmap="turbo", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(VAR_TEX[v], fontsize=12)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
        cax = fig.add_subplot(gs[n, j])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=8)
    fig.suptitle(f"{DS_PRETTY[ds]} — test trajectory {member}, $t=14$ ($t=0.7$)\n"
                 f"rows: ground truth, then each model under its best strategy "
                 f"(d = direct, r2 = 2-step, 662 = 6-6-2; label = avg. rel. $L_1$)", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fields_{ds}_m{member}.{ext}", dpi=150)
    plt.close(fig)
    return f"fields_{ds}_m{member}"


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--members", nargs="+", type=int, default=[9760, 9761, 9762])
    args = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    all_stats = {mode: load_mode(mode, TAB) for mode in MODES}
    meta = load_meta(TAB)
    report = ["# Evaluation report — neural operators on KH / RC / RKH", "",
              "Every model on the 240 held-out test trajectories per dataset, in all three prediction strategies "
              "(direct, 2-step, 6-6-2). All numbers are relative $L_1$ in **percent**.", ""]

    # ---- 1. per-timestep tables ----
    report += ["## 1. Average error across variables per timestep", ""]
    for mode in MODES:
        if all_stats[mode]:
            md, _ = per_timestep_tables(all_stats[mode], mode)
            report += [md, ""]

    # ---- 2. per-strategy final tables + best ----
    report += ["## 2. Final timestep ($t=14$, i.e. $t=0.7$): per-variable errors", ""]
    strat_avg: dict = {}       # (model) -> {mode: avg%}
    for mode in MODES:
        st = all_stats[mode]
        if not st:
            continue
        entries = [(m, stem, ds, label, d, mode) for (m, stem, ds, label, d) in rows_for(st, 14)]
        for m, _, _, _, d, _ in entries:
            strat_avg.setdefault(m, {})[mode] = 100 * np.mean([d[v].mean() for v in VARS])
        md = final_table(entries, f"$t=14$ — {MODES[mode][1]}", f"final_t14_{mode}",
                         caption=f"Per-variable relative $L_1$ errors at final time step ($t=0.7$), {MODES[mode][1]}.")
        report += [md, ""]
    best_strat = {m: min(d, key=d.get) for m, d in strat_avg.items()}
    best_entries = []
    for (m, stem, ds, label, _) in rows_for(all_stats["direct"] or all_stats["rollout662"], 14):
        bs = best_strat[m]
        best_entries.append((m, stem, ds, label, all_stats[bs][(m, 14)], bs))
    md = final_table(best_entries, "$t=14$ — best of {direct, 2-step, 6-6-2} per model × dataset", "final_t14_best",
                     note="The strategy column says which schedule gave the lowest average; the same per-trajectory "
                          "errors feed every column. ± = 95% bootstrap CI of the mean over 240 trajectories.")
    report += [md, "", "LaTeX version: `tables/final_t14_best.tex`.", ""]

    # ---- 3. strategy comparison ----
    cmp_md = ["### Average at $t=14$ by strategy", "",
              "| Model | Data | direct | 2-step | 6-6-2 | best |", "|---|---|---|---|---|---|"]
    cmp_rows = []
    for (m, stem, ds, label, _) in rows_for(all_stats["direct"] or all_stats["rollout662"], 14):
        d = strat_avg.get(m, {})
        bs = best_strat.get(m)
        cells = [f"**{d[k]:.2f}**" if k == bs else f"{d[k]:.2f}" if k in d else "--" for k in MODES]
        cmp_md.append(f"| {label} | {DS_PRETTY[ds]} | " + " | ".join(cells) + f" | {TAG.get(bs, '--')} |")
        cmp_rows.append({"model": label, "model_id": m, "dataset": DS_PRETTY[ds],
                         **{k: f"{d[k]:.4f}" if k in d else "" for k in MODES}, "best": bs or ""})
    _write_csv(TAB / "strategy_compare_t14.csv", cmp_rows)
    (TAB / "strategy_compare_t14.md").write_text("\n".join(cmp_md) + "\n")
    report += ["## 3. Strategy comparison", "", "\n".join(cmp_md), ""]

    # ---- 5. figures ----
    if not args.no_figures:
        fig_per_timestep(all_stats)
        made = []
        for ds in DATASETS:
            for mem in args.members:
                r = fig_fields(ds, mem, best_strat)
                if r:
                    made.append(r)
        report += ["## 4. Figures", "", "- `figures/per_timestep_curves.{pdf,png}` — average error vs timestep, "
                   "direct (top) and 2-step (bottom), one panel per dataset."]
        report += [f"- `figures/{r}.{{pdf,png}}`" for r in made]
        report.append("")

    # ---- 6. normalization audit + metadata ----
    nm = TAB / "normalization_check.md"
    if nm.exists():
        report += ["## 5. Normalization audit", "", nm.read_text(), ""]
    report += ["## 6. Run metadata", "", "| model | mode | checkpoint | epoch | step | params | GPU | det. | commit | wall (s) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(meta.items()):
        report.append(f"| {k.split('@')[0]} | {v.get('mode')} | `{Path(v['checkpoint']).parent.name}/{Path(v['checkpoint']).name}` | "
                      f"{v.get('trained_epoch')} | {v.get('trained_step')} | {v.get('params', 0) / 1e6:.1f}M | {v.get('gpu', '?')} | "
                      f"{v.get('deterministic')} | {v.get('git_commit')} | {v.get('wallclock_s')} |")
    (RES / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"[report] wrote {RES / 'REPORT.md'}; models with t=14 rows: {len(best_entries)}")


if __name__ == "__main__":
    main()
