"""Project-page assets that are derived from the results, so the page never drifts from them.

    python scripts/website/make_site_assets.py --fields-dir $PHAEDRA_OUTPUT_ROOT/evaluation/fields

1. `docs/static/images/operators/compare_<ds>.jpg` -- ground truth and every operator at t = 0.7
   for one test trajectory (4 variables), each model under its best strategy. Needs the dumps of
   `python -m evaluation.verify_dump_fields` (skipped with --no-figures).
2. The result tables and the per-timestep chart data of `docs/index.html`, generated from
   `docs/results/*.csv` and written between the `<!-- BEGIN:<name> -->` / `<!-- END:<name> -->`
   markers of the page.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "results"
PAGE = REPO / "docs" / "index.html"
IMAGES = REPO / "docs" / "static" / "images" / "operators"

VARS = ["rho", "u", "v", "p"]
VAR_LABEL = {"rho": "ρ", "u": "u", "v": "v", "p": "p"}
DATASETS = ["kh", "rc", "rkh"]
DS_LABEL = {"kh": "KH", "rc": "RC", "rkh": "RKH"}
# page order: token models, continuous latents, physical-space models
MODELS = [("phaedra_38m", "Phaedra", "Phaedra tokens"), ("fsq_38m", "FSQ", "FSQ tokens"),
          ("vqvae2_38m", "VQ-VAE-2", "VQ-VAE-2 tokens"), ("continuous_38m", "Continuous", "continuous AE latents"),
          ("fno_38m", "FNO", "physical fields"), ("cno_38m", "CNO", "physical fields"), ("vit_38m", "ViT", "physical fields")]
STRAT = {"direct": "direct", "rollout2": "2-step", "rollout662": "6-6-2"}


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open() as fh:
        return list(csv.DictReader(fh))


# ----------------------------------------------------------------------------- figures
def comparison_figure(fields_dir: Path, ds: str, member: int, best: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(fields_dir / f"fields_{ds}.npz")
    info = json.loads((fields_dir / f"fields_{ds}.json").read_text())
    gt = z[f"gt|t14|{member}"]
    cols = [("Ground truth", None, gt)]
    for stem, label, _ in MODELS:
        mid = f"{stem}_{ds}"
        strat = best[mid]
        e = info["relL1"][f"{mid}|{strat}|{member}"]
        avg = 100 * np.mean([e[v] for v in VARS])
        cols.append((label, f"{STRAT[strat]} · {avg:.1f}%", z[f"pred|{mid}|{strat}|{member}"]))
    n = len(cols)
    fig, axes = plt.subplots(len(VARS), n, figsize=(1.9 * n + 0.5, 1.9 * len(VARS) + 0.75), dpi=110,
                             gridspec_kw=dict(wspace=0.04, hspace=0.05, left=0.035, right=0.995, top=0.905, bottom=0.01))
    for i, v in enumerate(VARS):
        lo, hi = float(gt[i].min()), float(gt[i].max())
        for j, (label, sub, arr) in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(arr[i].T, origin="lower", cmap="turbo", vmin=lo, vmax=hi, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
                s.set_linewidth(0.5)
            if i == 0:
                ax.set_title(label + ("" if sub is None else f"\n{sub}"), fontsize=10.5,
                             fontweight="bold" if label in ("Ground truth", "Phaedra") else "normal", pad=5)
            if j == 0:
                ax.set_ylabel(VAR_LABEL[v], fontsize=13, rotation=0, labelpad=12, va="center")
    path = IMAGES / f"compare_{ds}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- html
def final_tables() -> tuple[str, str]:
    rows = {r["model_id"]: r for r in read_csv("final_t14_best.csv")}
    best = {ds: min((float(rows[f"{s}_{ds}"]["average"]), f"{s}_{ds}") for s, _, _ in MODELS)[1] for ds in DATASETS}
    tag = {"direct": "d", "rollout2": "2s", "rollout662": "662"}
    out = ['<table class="table is-bordered is-hoverable is-fullwidth results-table">',
           "<thead><tr><th>Model</th><th>Input</th>"
           + "".join(f'<th class="has-text-centered">{DS_LABEL[ds]}</th>' for ds in DATASETS) + "</tr></thead>", "<tbody>"]
    for stem, label, inp in MODELS:
        cls = ' class="phaedra-row"' if stem == "phaedra_38m" else ""
        cells = []
        for ds in DATASETS:
            r = rows[f"{stem}_{ds}"]
            txt = f'{float(r["average"]):.2f} <span class="ci">± {float(r["ci95_half"]):.2f}</span>'
            if best[ds] == f"{stem}_{ds}":
                txt = f"<strong>{txt}</strong>"
            cells.append(f'<td class="has-text-centered">{txt} <span class="strategy" title="{STRAT[r["strategy"]]}">'
                         f'{tag[r["strategy"]]}</span></td>')
        out.append(f"<tr{cls}><td>{label}{' transformer' if stem.startswith(('phaedra', 'fsq', 'vqvae2', 'continuous')) else ''}"
                   f"</td><td>{inp}</td>{''.join(cells)}</tr>")
    out += ["</tbody>", "</table>"]
    compact = "\n".join(out)

    full = ['<table class="table is-bordered is-hoverable is-fullwidth is-narrow results-table">',
            "<thead><tr><th>Model</th><th>Data</th>" + "".join(f'<th class="has-text-centered">{VAR_LABEL[v]}</th>' for v in VARS)
            + '<th class="has-text-centered">Average</th><th class="has-text-centered">Strategy</th></tr></thead>', "<tbody>"]
    for stem, label, _ in MODELS:
        for ds in DATASETS:
            r = rows[f"{stem}_{ds}"]
            cls = ' class="phaedra-row"' if stem == "phaedra_38m" else ""
            full.append(f"<tr{cls}><td>{label}</td><td>{DS_LABEL[ds]}</td>"
                        + "".join(f'<td class="has-text-centered">{float(r[v]):.2f}</td>' for v in VARS)
                        + f'<td class="has-text-centered">{float(r["average"]):.2f} ± {float(r["ci95_half"]):.2f}</td>'
                        f'<td class="has-text-centered">{STRAT[r["strategy"]]}</td></tr>')
    full += ["</tbody>", "</table>"]
    return compact, "\n".join(full)


def floor_table() -> str:
    rows = read_csv("tokenizer_floor_t14.csv")
    val = {(r["tokenizer"], r["dataset"]): float(r["average"]) for r in rows}
    names = ["Phaedra", "FSQ", "VQ-VAE-2", "Continuous AE"]
    out = ['<table class="table is-bordered is-hoverable is-narrow results-table floor-table">',
           "<thead><tr><th>Tokenizer</th>" + "".join(f'<th class="has-text-centered">{DS_LABEL[d]}</th>' for d in DATASETS)
           + "</tr></thead>", "<tbody>"]
    for n in names:
        cls = ' class="phaedra-row"' if n == "Phaedra" else ""
        note = ' <span class="ci">(not discrete)</span>' if n == "Continuous AE" else ""
        out.append(f"<tr{cls}><td>{n}{note}</td>" + "".join(
            f'<td class="has-text-centered">{val[(n, DS_LABEL[d])]:.2f}</td>' for d in DATASETS) + "</tr>")
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def mae_table() -> str:
    rows = {r["model_id"]: r for r in read_csv("mae.csv")}
    cols = [("3pde", "RKH", "pre-trained on 3 PDEs"), ("finetune_kh", "KH", "fine-tuned"), ("finetune_rc", "RC", "fine-tuned"),
            ("finetune_rkh", "RKH", "fine-tuned")]
    out = ['<table class="table is-bordered is-hoverable is-narrow is-fullwidth results-table">',
           "<thead><tr><th>Tokens</th>" + "".join(f'<th class="has-text-centered">{h}<br><span class="ci">evaluated on {d}</span></th>'
                                                  for _, d, h in cols) + "</tr></thead>", "<tbody>"]
    for tok, label in (("phaedra", "Phaedra"), ("fsq", "FSQ")):
        cls = ' class="phaedra-row"' if tok == "phaedra" else ""
        cells = []
        for c, _, _ in cols:
            r = rows[f"mae_{tok}_{c}"]
            n = int(r["test_trajectories"])
            cells.append(f'<td class="has-text-centered">{float(r["average"]):.2f}'
                         f'<span class="ci"> ({n} traj.)</span></td>')
        out.append(f"<tr{cls}><td>{label}</td>{''.join(cells)}</tr>")
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def chart_json() -> str:
    data = {"datasets": [DS_LABEL[d] for d in DATASETS], "strategies": {}, "models": [m[1] for m in MODELS]}
    for mode in ("direct", "rollout2", "rollout662"):
        rows = read_csv(f"per_timestep_{mode}.csv")
        ts = [int(k[1:]) for k in rows[0] if re.fullmatch(r"t\d+", k)]
        series = {}
        for r in rows:
            stem = r["model_id"].rsplit("_", 1)[0]
            label = next(l for s, l, _ in MODELS if s == stem)
            series.setdefault(r["dataset"], {})[label] = [round(float(r[f"t{t}"]), 3) if r[f"t{t}"] else None for t in ts]
        data["strategies"][STRAT[mode]] = {"t": [round(0.05 * t, 2) for t in ts], "series": series}
    return '<script id="results-data" type="application/json">' + json.dumps(data, separators=(",", ":")) + "</script>"


def inject(page: str, name: str, content: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    if not pat.search(page):
        raise SystemExit(f"marker {name} not found in {PAGE}")
    return pat.sub(lambda m: m.group(1) + "\n" + content + "\n" + m.group(3), page)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-dir", default=None, help="dumps of evaluation.verify_dump_fields")
    ap.add_argument("--member", type=int, default=9760)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if not args.no_figures:
        best = {r["model_id"]: r["strategy"] for r in read_csv("final_t14_best.csv")}
        for ds in DATASETS:
            p = comparison_figure(Path(args.fields_dir), ds, args.member, best)
            print(f"[site] {p.relative_to(REPO)} ({p.stat().st_size / 1e3:.0f} kB)")
    page = PAGE.read_text()
    compact, full = final_tables()
    for name, content in (("final-table", compact), ("full-table", full), ("floor-table", floor_table()),
                          ("mae-table", mae_table()), ("results-json", chart_json())):
        page = inject(page, name, content)
    PAGE.write_text(page)
    print(f"[site] tables and chart data -> {PAGE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
