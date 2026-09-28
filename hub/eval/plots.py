from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


_LATEX_STYLE_RCPARAMS = {
    # Prefer Computer Modern style while keeping usetex disabled for portability.
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.size": 12,
}


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _style_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_token_plot(path: str, target_morph: torch.Tensor, pred_morph: torch.Tensor, target_amp: torch.Tensor, pred_amp: torch.Tensor) -> None:
    # Shapes: [V, H, W]
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    v = int(target_morph.shape[0])
    with plt.rc_context(_LATEX_STYLE_RCPARAMS):
        fig, axes = plt.subplots(v, 4, figsize=(12, 2.8 * v), squeeze=False, constrained_layout=True)
        for i in range(v):
            panels = [
                (target_morph[i], "Ground Truth Morphology", "tab20"),
                (pred_morph[i], "Predicted Morphology", "tab20"),
                (target_amp[i], "Ground Truth Amplitude", "viridis"),
                (pred_amp[i], "Predicted Amplitude", "viridis"),
            ]
            for j, (arr, title, cmap) in enumerate(panels):
                im = axes[i][j].imshow(_to_np(arr), cmap=cmap)
                if i == 0:
                    axes[i][j].set_title(title, fontsize=14, pad=7)
                _style_axis(axes[i][j])
                fig.colorbar(im, ax=axes[i][j], fraction=0.046, pad=0.04)
        fig.savefig(path_obj, dpi=140)
        plt.close(fig)


def save_field_plot(path: str, target_field: torch.Tensor, pred_field: torch.Tensor) -> None:
    # Shapes: [V, H, W]
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    v = int(target_field.shape[0])
    with plt.rc_context(_LATEX_STYLE_RCPARAMS):
        fig, axes = plt.subplots(
            v,
            5,
            figsize=(12, max(2.8 * v, 3.2)),
            squeeze=False,
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.06, 0.06]},
        )
        for i in range(v):
            diff = pred_field[i] - target_field[i]
            field_vmin, field_vmax = _pair_limits(target_field[i], pred_field[i])
            diff_abs_max = float(diff.abs().max().item())
            if np.isclose(diff_abs_max, 0.0):
                diff_abs_max = 1e-8

            gt_im = axes[i][0].imshow(_to_np(target_field[i]), cmap="turbo", vmin=field_vmin, vmax=field_vmax)
            pred_im = axes[i][1].imshow(_to_np(pred_field[i]), cmap="turbo", vmin=field_vmin, vmax=field_vmax)
            diff_im = axes[i][2].imshow(_to_np(diff), cmap="bwr", vmin=-diff_abs_max, vmax=diff_abs_max)

            if i == 0:
                axes[i][0].set_title("Ground Truth", fontsize=14, pad=7)
                axes[i][1].set_title("Prediction", fontsize=14, pad=7)
                axes[i][2].set_title("Difference", fontsize=14, pad=7)
            _style_axis(axes[i][0])
            _style_axis(axes[i][1])
            _style_axis(axes[i][2])
            _style_axis(axes[i][3])
            _style_axis(axes[i][4])

            # Dedicated colorbar columns prevent overlap with image panels.
            cb_field = fig.colorbar(gt_im, cax=axes[i][3])
            cb_field.ax.tick_params(labelsize=10)
            cb_diff = fig.colorbar(diff_im, cax=axes[i][4])
            cb_diff.ax.tick_params(labelsize=10)

            if i == 0:
                cb_field.set_label("Field Value", fontsize=12)
                cb_diff.set_label("Prediction - Ground Truth", fontsize=12)

        fig.savefig(path_obj, dpi=140)
        plt.close(fig)


def save_paper_triplet_plot(
    path: str,
    ground_truth_field: torch.Tensor,
    recon_from_gt_tokens: torch.Tensor,
    recon_from_pred_tokens: torch.Tensor,
) -> None:
    """Save a minimalist paper figure with rows=variables and columns=3.

    Column order per row: ground-truth field, reconstruction from GT tokens,
    reconstruction from predicted tokens. No axes, titles, captions, or colorbars.
    """
    if ground_truth_field.shape != recon_from_gt_tokens.shape or ground_truth_field.shape != recon_from_pred_tokens.shape:
        raise ValueError(
            "All paper triplet tensors must have identical shape [V, H, W], got: "
            f"gt={tuple(ground_truth_field.shape)}, "
            f"gt_recon={tuple(recon_from_gt_tokens.shape)}, "
            f"pred_recon={tuple(recon_from_pred_tokens.shape)}"
        )

    if ground_truth_field.ndim != 3:
        raise ValueError(f"paper triplet tensors must be rank-3 [V, H, W], got ndim={ground_truth_field.ndim}")

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    n_vars = int(ground_truth_field.shape[0])

    with plt.rc_context(_LATEX_STYLE_RCPARAMS):
        fig, axes = plt.subplots(
            n_vars,
            3,
            figsize=(6, max(2.0, 2.0 * n_vars)),
            squeeze=False,
        )

        for i in range(n_vars):
            row_vmin = min(
                float(ground_truth_field[i].min().item()),
                float(recon_from_gt_tokens[i].min().item()),
                float(recon_from_pred_tokens[i].min().item()),
            )
            row_vmax = max(
                float(ground_truth_field[i].max().item()),
                float(recon_from_gt_tokens[i].max().item()),
                float(recon_from_pred_tokens[i].max().item()),
            )
            if np.isclose(row_vmin, row_vmax):
                pad = 1.0 if abs(row_vmin) < 1e-8 else max(1e-8, 0.01 * abs(row_vmin))
                row_vmin -= pad
                row_vmax += pad

            axes[i][0].imshow(_to_np(ground_truth_field[i]), cmap="turbo", vmin=row_vmin, vmax=row_vmax)
            axes[i][1].imshow(_to_np(recon_from_gt_tokens[i]), cmap="turbo", vmin=row_vmin, vmax=row_vmax)
            axes[i][2].imshow(_to_np(recon_from_pred_tokens[i]), cmap="turbo", vmin=row_vmin, vmax=row_vmax)

            _style_axis(axes[i][0])
            _style_axis(axes[i][1])
            _style_axis(axes[i][2])

        # Very tight layout for paper-ready panels.
        fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.01, hspace=0.01)
        fig.savefig(path_obj, dpi=220, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


def _pair_limits(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    vmin = min(float(a.min().item()), float(b.min().item()))
    vmax = max(float(a.max().item()), float(b.max().item()))
    if np.isclose(vmin, vmax):
        pad = 1.0 if abs(vmin) < 1e-8 else max(1e-8, 0.01 * abs(vmin))
        return vmin - pad, vmax + pad
    return vmin, vmax


def save_sample_overview_plot(
    path: str,
    target_morph: torch.Tensor,
    pred_morph: torch.Tensor,
    target_amp: torch.Tensor,
    pred_amp: torch.Tensor,
    target_field: torch.Tensor,
    pred_field: torch.Tensor,
    var_names: list[str] | None = None,
) -> None:
    """Save one figure per sample with GT/Pred for morph, amp, and reconstruction.

    Each GT/pred pair must have matching [V, H, W] shapes. Spatial resolutions
    can differ between morphology, amplitude, and reconstruction.
    For each variable, GT and prediction panels for the same quantity share
    the same vmin/vmax.
    """
    if target_morph.shape != pred_morph.shape:
        raise ValueError(
            "Morphology GT/pred shape mismatch: "
            f"{tuple(target_morph.shape)} vs {tuple(pred_morph.shape)}"
        )
    if target_amp.shape != pred_amp.shape:
        raise ValueError(
            "Amplitude GT/pred shape mismatch: "
            f"{tuple(target_amp.shape)} vs {tuple(pred_amp.shape)}"
        )
    if target_field.shape != pred_field.shape:
        raise ValueError(
            "Reconstruction GT/pred shape mismatch: "
            f"{tuple(target_field.shape)} vs {tuple(pred_field.shape)}"
        )

    if target_morph.ndim != 3 or target_amp.ndim != 3 or target_field.ndim != 3:
        raise ValueError("All overview tensors must be rank-3 [V, H, W]")

    n_vars = int(target_morph.shape[0])
    if int(target_amp.shape[0]) != n_vars or int(target_field.shape[0]) != n_vars:
        raise ValueError(
            "Variable dimension mismatch across modalities: "
            f"morph={int(target_morph.shape[0])}, "
            f"amp={int(target_amp.shape[0])}, "
            f"recon={int(target_field.shape[0])}"
        )

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(_LATEX_STYLE_RCPARAMS):
        fig, axes = plt.subplots(
            n_vars,
            9,
            figsize=(20, max(3.0, 2.8 * n_vars)),
            squeeze=False,
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 1.0, 0.06, 1.0, 1.0, 0.06, 1.0, 1.0, 0.06]},
        )

        for i in range(n_vars):
            morph_vmin, morph_vmax = _pair_limits(target_morph[i], pred_morph[i])
            amp_vmin, amp_vmax = _pair_limits(target_amp[i], pred_amp[i])
            field_vmin, field_vmax = _pair_limits(target_field[i], pred_field[i])

            morph_gt = axes[i][0].imshow(_to_np(target_morph[i]), cmap="nipy_spectral", vmin=morph_vmin, vmax=morph_vmax)
            morph_pr = axes[i][1].imshow(_to_np(pred_morph[i]), cmap="nipy_spectral", vmin=morph_vmin, vmax=morph_vmax)
            amp_gt = axes[i][3].imshow(_to_np(target_amp[i]), cmap="viridis", vmin=amp_vmin, vmax=amp_vmax)
            amp_pr = axes[i][4].imshow(_to_np(pred_amp[i]), cmap="viridis", vmin=amp_vmin, vmax=amp_vmax)
            rec_gt = axes[i][6].imshow(_to_np(target_field[i]), cmap="turbo", vmin=field_vmin, vmax=field_vmax)
            rec_pr = axes[i][7].imshow(_to_np(pred_field[i]), cmap="turbo", vmin=field_vmin, vmax=field_vmax)

            if i == 0:
                axes[i][0].set_title("Ground Truth Morphology", fontsize=14, pad=7)
                axes[i][1].set_title("Predicted Morphology", fontsize=14, pad=7)
                axes[i][3].set_title("Ground Truth Amplitude", fontsize=14, pad=7)
                axes[i][4].set_title("Predicted Amplitude", fontsize=14, pad=7)
                axes[i][6].set_title("Ground Truth Reconstruction", fontsize=14, pad=7)
                axes[i][7].set_title("Predicted Reconstruction", fontsize=14, pad=7)

            for ax_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                _style_axis(axes[i][ax_idx])

            if var_names is not None and i < len(var_names):
                axes[i][0].set_ylabel(str(var_names[i]), fontsize=12)
            else:
                axes[i][0].set_ylabel(f"v{i}", fontsize=12)

            cb_morph = fig.colorbar(morph_pr, cax=axes[i][2])
            cb_amp = fig.colorbar(amp_pr, cax=axes[i][5])
            cb_recon = fig.colorbar(rec_pr, cax=axes[i][8])
            cb_morph.ax.tick_params(labelsize=10)
            cb_amp.ax.tick_params(labelsize=10)
            cb_recon.ax.tick_params(labelsize=10)

            if i == 0:
                cb_morph.set_label("Morphology", fontsize=12)
                cb_amp.set_label("Amplitude", fontsize=12)
                cb_recon.set_label("Reconstruction", fontsize=12)

        fig.savefig(path_obj, dpi=140)
        plt.close(fig)


def save_field_overview_plot(
    path: str,
    target_fields: list[torch.Tensor],
    pred_fields: list[torch.Tensor],
    var_names: list[str] | None = None,
) -> None:
    """Save all reconstructions in a single figure with shared color ranges.

    Each entry in target_fields/pred_fields has shape [V, H, W].
    """
    if torch.is_tensor(target_fields):
        target_fields = [target_fields]
    if torch.is_tensor(pred_fields):
        pred_fields = [pred_fields]
    if len(target_fields) == 0 or len(pred_fields) == 0:
        return
    if len(target_fields) != len(pred_fields):
        raise ValueError("target_fields and pred_fields must have the same length")

    n_samples = len(target_fields)
    n_vars = int(target_fields[0].shape[0])
    for idx in range(n_samples):
        if target_fields[idx].shape != pred_fields[idx].shape:
            raise ValueError(f"target/pred shape mismatch at sample {idx}")

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    field_min = min(float(t.min().item()) for t in target_fields + pred_fields)
    field_max = max(float(t.max().item()) for t in target_fields + pred_fields)
    diff_abs_max = max(float((p - t).abs().max().item()) for t, p in zip(target_fields, pred_fields))
    diff_abs_max = max(diff_abs_max, 1e-8)

    n_rows = n_samples * n_vars
    with plt.rc_context(_LATEX_STYLE_RCPARAMS):
        fig, axes = plt.subplots(
            n_rows,
            5,
            figsize=(12, max(3, int(2.2 * n_rows))),
            squeeze=False,
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.06, 0.06]},
        )

        for s in range(n_samples):
            tgt = target_fields[s]
            pred = pred_fields[s]
            for v in range(n_vars):
                row = s * n_vars + v
                diff = pred[v] - tgt[v]

                gt_im = axes[row][0].imshow(_to_np(tgt[v]), cmap="turbo", vmin=field_min, vmax=field_max)
                pr_im = axes[row][1].imshow(_to_np(pred[v]), cmap="turbo", vmin=field_min, vmax=field_max)
                df_im = axes[row][2].imshow(_to_np(diff), cmap="bwr", vmin=-diff_abs_max, vmax=diff_abs_max)

                if row == 0:
                    axes[row][0].set_title("Ground Truth", fontsize=14, pad=7)
                    axes[row][1].set_title("Prediction", fontsize=14, pad=7)
                    axes[row][2].set_title("Difference", fontsize=14, pad=7)

                if var_names is not None and v < len(var_names):
                    axes[row][0].set_ylabel(f"s{s} {var_names[v]}", fontsize=12)
                else:
                    axes[row][0].set_ylabel(f"s{s} v{v}", fontsize=12)

                _style_axis(axes[row][0])
                _style_axis(axes[row][1])
                _style_axis(axes[row][2])
                _style_axis(axes[row][3])
                _style_axis(axes[row][4])

                cb_field = fig.colorbar(pr_im, cax=axes[row][3])
                cb_diff = fig.colorbar(df_im, cax=axes[row][4])
                cb_field.ax.tick_params(labelsize=10)
                cb_diff.ax.tick_params(labelsize=10)
                if row == 0:
                    cb_field.set_label("Field Value", fontsize=12)
                    cb_diff.set_label("Prediction - Ground Truth", fontsize=12)

        fig.savefig(path_obj, dpi=140)
        plt.close(fig)
