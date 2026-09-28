"""Project-page video: the ground truth, the Phaedra tokens a neural operator predicts, and the
field decoded from those tokens, side by side for one test trajectory of each dataset.

    # predict (GPU, released weights under $PHAEDRA_OUTPUT_ROOT) and render
    python scripts/website/render_token_video.py --deterministic --out-dir docs/static/videos
    # re-render from the saved predictions (CPU only)
    python scripts/website/render_token_video.py --from-npz docs/static/videos/token_rollout.npz

Every frame t in {0, 2, ..., 14} (physical time 0.05 t) is predicted DIRECTLY from the tokens of the
initial condition -- one call of the operator with lead time t, the `direct` strategy of
`evaluation.eval_downstream` -- and decoded by the frozen Phaedra tokenizer. The t = 0 frame shows the
input tokens and their reconstruction. `--mode rollout2` chains 2-step predictions instead.
With `--check-csv-dir` the per-frame errors are compared with the per-trajectory CSVs written by
`evaluation.eval_downstream` (identical under --deterministic on the same GPU model).

Needs imageio + imageio-ffmpeg (pip install imageio imageio-ffmpeg) for the mp4.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODELS = ["phaedra_38m_kh", "phaedra_38m_rc", "phaedra_38m_rkh"]
TIMES = list(range(0, 15, 2))            # time indices; physical time = 0.05 * index
DT = 0.05
FSQ_LEVELS = np.array([5, 4, 4, 3, 3, 3, 2, 2])     # Phaedra morphology codebook: 8640 codes
DATASET_NAME = {"kh": "Kelvin–Helmholtz", "rc": "curved Riemann", "rkh": "Riemann–\nKelvin–Helmholtz"}
VAR_NAME = {"rho": "density ρ", "u": "velocity u", "v": "velocity v", "p": "pressure p"}


# ----------------------------------------------------------------------------- prediction (GPU)
def predict(models: list[str], member_rank: int, mode: str, deterministic: bool) -> dict:
    import netCDF4 as nc
    import torch
    from omegaconf import OmegaConf

    from evaluation.eval_downstream import FAMILIES, REGISTRY, _test_members
    from fno_operator.src.dataset import read_fields
    from hub.trainers.test import _load_member_tokens_phaedra, _relative_l1
    from hub.utils.runtime import configure_torch, get_device

    if deterministic:                     # same switches as eval_downstream --deterministic
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        configure_torch(tf32=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)
        np.random.seed(0)
    else:
        configure_torch(tf32=True)
    device = get_device()
    reg = OmegaConf.to_container(OmegaConf.load(REGISTRY), resolve=True)
    out = {"mode": mode, "times": TIMES, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "deterministic": deterministic, "models": {}}
    for name in models:
        entry = reg["models"][name]
        if entry["family"] != "hub_phaedra":
            raise SystemExit(f"{name}: only operators on Phaedra tokens have amplitude/morphology tokens")
        ds_cfg = reg["datasets"][entry["dataset"]]
        runner = FAMILIES[entry["family"]](entry, ds_cfg, reg["decoders"], device)
        idx, ids = _test_members(runner.token_path)
        mi, mid = idx[member_rank], ids[member_rank]
        src = nc.Dataset(ds_cfg["source_fields"], "r")
        if "member" in src.variables:
            lookup = {int(m): i for i, m in enumerate(np.asarray(src.variables["member"][:], dtype=np.int64))}
            si = lookup.get(mid, mid)
        else:
            si = mid
        start = runner.make_start_state(mi)
        rec = {k: [] for k in ("amp", "morph", "true_amp", "true_morph", "pred", "true")}
        rel = []
        state = start
        for t in TIMES:
            if t > 0:
                state = runner.step_state(start, 0, t) if mode == "direct" else runner.step_state(state, t - 2, t)
            tok = start if t == 0 else state
            fields = runner.decode_state(tok)
            gt = read_fields(src, member_index=si, time_idx=t, variables=runner.variables)
            ta, tm = _load_member_tokens_phaedra(runner._ds, mi, t, runner.variables, runner.morph_offset)
            rec["amp"].append(tok[0].numpy().astype(np.int16))
            rec["morph"].append(tok[1].numpy().astype(np.int16))
            rec["true_amp"].append(ta.numpy().astype(np.int16))
            rec["true_morph"].append(tm.numpy().astype(np.int16))
            rec["pred"].append(np.stack([fields[v] for v in runner.variables]).astype(np.float32))
            rec["true"].append(gt.astype(np.float32))
            rel.append([_relative_l1(fields[v], gt[i]) for i, v in enumerate(runner.variables)])
        src.close()
        runner.close()
        out["models"][name] = {"dataset": entry["dataset"], "member": int(mid), "variables": list(runner.variables),
                               **{k: np.stack(v) for k, v in rec.items()}, "relL1": np.array(rel)}
        r14 = dict(zip(runner.variables, np.round(100 * out["models"][name]["relL1"][-1], 2)))
        print(f"[video] {name}: test member {mid}, rel. L1 at t=0.7 (%): {r14}", flush=True)
    return out


def save_npz(data: dict, path: Path) -> None:
    arrays = {"meta": np.array(json.dumps({k: v for k, v in data.items() if k != "models"} |
                                          {"order": list(data["models"])}))}
    for name, m in data["models"].items():
        arrays[f"{name}|info"] = np.array(json.dumps({"dataset": m["dataset"], "member": m["member"],
                                                      "variables": m["variables"]}))
        for k in ("amp", "morph", "true_amp", "true_morph", "pred", "true", "relL1"):
            arrays[f"{name}|{k}"] = m[k]
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    data = {k: v for k, v in meta.items() if k != "order"} | {"models": {}}
    for name in meta["order"]:
        m = json.loads(str(z[f"{name}|info"]))
        for k in ("amp", "morph", "true_amp", "true_morph", "pred", "true", "relL1"):
            m[k] = z[f"{name}|{k}"]
        data["models"][name] = m
    return data


def check_against_csv(data: dict, csv_dir: Path) -> bool:
    """Frames t >= 2 vs the per-trajectory errors of evaluation.eval_downstream (same mode)."""
    stem = {"direct": "per_sample_relL1", "rollout2": "per_sample_relL1_rollout"}[data["mode"]]
    ok = True
    for name, m in data["models"].items():
        f = csv_dir / f"{stem}.{name}.csv"
        if not f.exists():
            f = csv_dir / f"{stem}.csv"
        with f.open() as fh:
            ref = {(r["var"], int(r["t"])): r["relL1"] for r in csv.DictReader(fh)
                   if r["model"] == name and int(r["traj_id"]) == m["member"]}
        n = same = 0
        for ti, t in enumerate(data["times"]):
            for vi, v in enumerate(m["variables"]):
                if (v, t) in ref:
                    n += 1
                    same += f"{m['relL1'][ti, vi]:.8f}" == ref[(v, t)]
        print(f"[check] {name}: {same}/{n} per-frame errors identical to {f.name}", flush=True)
        ok &= n > 0 and same == n
    return ok


# ----------------------------------------------------------------------------- rendering (CPU)
def morph_digits(ids: np.ndarray) -> np.ndarray:
    """FSQ index -> its code digits scaled to [0, 1] (8 dims)."""
    basis = np.cumprod(np.concatenate([[1], FSQ_LEVELS[:-1]]))
    return ((ids[..., None].astype(np.int64) // basis) % FSQ_LEVELS) / (FSQ_LEVELS - 1)


def morph_colouring(data: dict):
    """One fixed map from morphology codes to RGB for the whole video: PCA of the FSQ code
    digits of every token shown, first three components -> RGB."""
    ids = np.concatenate([np.concatenate([m["morph"].ravel(), m["true_morph"].ravel()])
                          for m in data["models"].values()])
    x = morph_digits(ids)
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    comp = vt[:3]
    comp *= np.sign(comp[np.arange(3), np.abs(comp).argmax(1)])[:, None]
    y = (x - mu) @ comp.T
    lo, hi = np.percentile(y, 1, axis=0), np.percentile(y, 99, axis=0)

    def rgb(grid: np.ndarray) -> np.ndarray:
        return np.clip(((morph_digits(grid) - mu) @ comp.T - lo) / (hi - lo), 0, 1)
    return rgb


def render_frames(data: dict, var: str, rgb) -> list[np.ndarray]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W, H, dpi = 1280, 880, 100
    names = list(data["models"])
    n_rows = len(names)
    label_w, pad, gap = 160, 16, 12
    top, bottom = 132, 66
    panel = min((W - label_w - pad - 4 * gap) // 5, (H - top - bottom - (n_rows - 1) * gap) // n_rows)
    x0 = label_w + (W - label_w - pad - 5 * panel - 4 * gap) // 2

    rows = []
    for name in names:
        m = data["models"][name]
        vi = m["variables"].index(var)
        true, pred = m["true"][:, vi], m["pred"][:, vi]
        err = np.abs(pred - true)
        amps = np.concatenate([m["amp"][:, vi].ravel(), m["true_amp"][:, vi].ravel()])
        rows.append({"m": m, "vi": vi, "true": true, "pred": pred, "err": err,
                     "f_lim": np.percentile(true, [0.5, 99.5]),
                     "e_max": max(float(np.percentile(err[1:], 99.5)), 1e-12),
                     "a_lim": np.percentile(amps, [1, 99])})

    frames = []
    for ti, t in enumerate(data["times"]):
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        fig.patch.set_facecolor("white")

        def ax_at(x, y, w, h):
            return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

        first = t == 0
        how = "input tokens of the initial condition" if first else (
            f"predicted from the tokens at t = 0 in one step (lead time {t * DT:.2f})" if data["mode"] == "direct"
            else f"autoregressive 2-step rollout, {t // 2} step{'s' if t > 2 else ''}")
        fig.text(pad / W, 1 - 30 / H, "Phaedra neural operator: predicted tokens and the physics decoded from them",
                 fontsize=16, fontweight="bold", va="center")
        fig.text(pad / W, 1 - 58 / H, f"{VAR_NAME[var]}  ·  first test trajectory of each dataset  ·  {how}",
                 fontsize=11.5, color="#444444", va="center")
        fig.text(1 - pad / W, 1 - 32 / H, f"t = {t * DT:.2f}", fontsize=22, fontweight="bold", ha="right", va="center",
                 color="#1f4e8c")
        heads = (["Ground truth\n(128 × 128)", "Amplitude tokens\n(input, 32 × 32)", "Morphology tokens\n(input, 32 × 32)",
                  "Decoded input\ntokens", "|decoded − truth|\n"] if first else
                 ["Ground truth\n(128 × 128)", "Predicted amplitude\ntokens (32 × 32)",
                  "Predicted morphology\ntokens (32 × 32)", "Decoded prediction\n(128 × 128)", "|prediction − truth|\n"])
        for c, h in enumerate(heads):
            fig.text((x0 + c * (panel + gap) + panel / 2) / W, 1 - (top - 24) / H, h, ha="center", va="center",
                     fontsize=11, fontweight="bold", color="#222222", linespacing=1.25)
        for r, row in enumerate(rows):
            y = top + r * (panel + gap)
            m = row["m"]
            fig.text((label_w - 12) / W, 1 - (y + panel / 2 - 12) / H, m["dataset"].upper(), ha="right", va="center",
                     fontsize=18, fontweight="bold")
            fig.text((label_w - 12) / W, 1 - (y + panel / 2 + 8) / H, DATASET_NAME[m["dataset"]], ha="right",
                     va="top", fontsize=10, color="#555555", linespacing=1.2)
            imgs = [(row["true"][ti], dict(cmap="turbo", vmin=row["f_lim"][0], vmax=row["f_lim"][1])),
                    (m["amp"][ti, row["vi"]], dict(cmap="viridis", vmin=row["a_lim"][0], vmax=row["a_lim"][1],
                                                   interpolation="nearest")),
                    (rgb(m["morph"][ti, row["vi"]]), dict(interpolation="nearest")),
                    (row["pred"][ti], dict(cmap="turbo", vmin=row["f_lim"][0], vmax=row["f_lim"][1])),
                    (row["err"][ti], dict(cmap="inferno", vmin=0, vmax=row["e_max"]))]
            for c, (img, kw) in enumerate(imgs):
                ax = ax_at(x0 + c * (panel + gap), y, panel, panel)
                ax.imshow(np.asarray(img).swapaxes(0, 1), origin="lower", **({"interpolation": "bilinear"} | kw))
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#999999")
                    s.set_linewidth(0.6)
                if c == 4:
                    ax.text(0.04, 0.05, f"rel. L$_1$ {100 * m['relL1'][ti, row['vi']]:.1f}%", transform=ax.transAxes,
                            color="white", fontsize=11, fontweight="bold", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, lw=0))
        # timeline
        ax = ax_at(x0, H - 44, 5 * panel + 4 * gap, 22)
        ts = np.array(data["times"]) * DT
        ax.plot([ts[0], ts[-1]], [0, 0], color="#cccccc", lw=3, zorder=1, solid_capstyle="round")
        ax.plot([ts[0], ts[ti]], [0, 0], color="#1f4e8c", lw=3, zorder=2, solid_capstyle="round")
        ax.scatter(ts, np.zeros_like(ts), s=36, zorder=3,
                   c=["#1f4e8c" if i <= ti else "#cccccc" for i in range(len(ts))])
        for i, tt in enumerate(ts):
            ax.text(tt, -1.5, f"{tt:.1f}", ha="center", va="top", fontsize=9, color="#555555")
        ax.set_xlim(ts[0] - 0.01, ts[-1] + 0.01)
        ax.set_ylim(-3, 1)
        ax.axis("off")
        fig.text(x0 / W - 0.005, 1 - (H - 38) / H, "time", ha="right", va="center", fontsize=9.5, color="#555555")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    return frames


def write_video(frames: list[np.ndarray], path: Path, fps: int = 10, hold=(12, 8, 20), crf: int = 22) -> None:
    import imageio.v2 as imageio
    first, mid, last = hold
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=16,
                            ffmpeg_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"]) as w:
        for i, f in enumerate(frames):
            for _ in range(first if i == 0 else last if i == len(frames) - 1 else mid):
                w.append_data(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--member-rank", type=int, default=0, help="index into each dataset's test split (0 = first)")
    ap.add_argument("--mode", choices=["direct", "rollout2"], default="direct")
    ap.add_argument("--variables", nargs="+", default=["rho", "u", "v", "p"])
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "static" / "videos"))
    ap.add_argument("--from-npz", default=None, help="re-render saved predictions (no GPU needed)")
    ap.add_argument("--npz", default=None, help="where to save the predictions (default: a scratch file in "
                                                "$PHAEDRA_OUTPUT_ROOT/evaluation/video)")
    ap.add_argument("--check-csv-dir", default=None, help="directory with per_sample_relL1*.csv to compare against")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--crf", type=int, default=22, help="x264 quality (lower = larger file)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_npz:
        data = load_npz(Path(args.from_npz))
    else:
        data = predict(args.models, args.member_rank, args.mode, args.deterministic)
        npz = Path(args.npz) if args.npz else (Path(os.environ.get("PHAEDRA_OUTPUT_ROOT", ".")) / "evaluation" /
                                                "video" / f"token_rollout_{args.mode}.npz")
        npz.parent.mkdir(parents=True, exist_ok=True)
        save_npz(data, npz)
        print(f"[video] predictions -> {npz}", flush=True)
    if args.check_csv_dir and not check_against_csv(data, Path(args.check_csv_dir)):
        raise SystemExit("[check] per-frame errors differ from the evaluation CSVs")

    from PIL import Image
    rgb = morph_colouring(data)
    summary = {"mode": data["mode"], "gpu": data.get("gpu"), "deterministic": data.get("deterministic"),
               "times": [round(t * DT, 2) for t in data["times"]], "videos": {}}
    for var in args.variables:
        frames = render_frames(data, var, rgb)
        mp4 = out / f"phaedra_tokens_{var}.mp4"
        write_video(frames, mp4, crf=args.crf)
        Image.fromarray(frames[-1]).save(out / f"phaedra_tokens_{var}_poster.jpg", quality=88, optimize=True)
        summary["videos"][var] = {"file": mp4.name, "bytes": mp4.stat().st_size,
                                  "relL1_percent": {name: [round(100 * float(x), 3) for x in
                                                           m["relL1"][:, m["variables"].index(var)]]
                                                    for name, m in data["models"].items()},
                                  "members": {name: m["member"] for name, m in data["models"].items()}}
        print(f"[video] {mp4} ({mp4.stat().st_size / 1e6:.2f} MB)", flush=True)
    (out / "phaedra_tokens.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
