from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf

from mae.src.model import MAEConfig, TokenMAE

REPO_ROOT = Path(__file__).resolve().parents[1]
from tokenizer import config_path as _tok_config  # noqa: E402

from tokenizer.systems import FSQAESystem, PhaedraAEFSQSystem


_LATEX_STYLE_RCPARAMS = {
    # Keep TeX-like styling without requiring an external TeX installation.
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 11,
}


@dataclass
class TokenizerHandle:
    task: object
    model_type: str
    device: torch.device


def _load_yaml(path: Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _load_state_dict_from_path(path: Path) -> dict:
    """Weights from a released `model.safetensors`, a training checkpoint, or a directory."""
    from hub.utils.weights import load_weights
    return load_weights(path)[0]


def _resolve_checkpoint_path(path: Path) -> Path:
    from hub.utils.weights import resolve_weights_file
    return resolve_weights_file(path)


def _maybe_apply_model_ema_from_checkpoint(model: torch.nn.Module, checkpoint_path: Path, use_ema: bool) -> bool:
    if not use_ema:
        print("[mae-test] MAE EMA disabled via config")
        return False
    raw_path = _resolve_checkpoint_path(checkpoint_path)
    if raw_path.suffix == ".safetensors":
        from hub.utils.weights import load_weights
        meta = load_weights(raw_path)[1]["metadata"]
        baked = meta.get("ema") == "applied"
        print(f"[mae-test] released weights ({'EMA' if baked else 'raw'} weights, as evaluated in the paper)")
        return baked
    raw_state = torch.load(raw_path, map_location="cpu", weights_only=False)
    if not isinstance(raw_state, dict) or "ema" not in raw_state:
        print("[mae-test] MAE checkpoint has no ema state; using non-EMA model weights")
        return False
    ema_state = raw_state["ema"]
    shadow = ema_state.get("shadow_params") if isinstance(ema_state, dict) else None
    if shadow is None:
        print("[mae-test] MAE ema state missing shadow_params; using non-EMA model weights")
        return False

    params = list(model.parameters())
    if len(params) != len(shadow):
        print("[mae-test] MAE ema parameter count mismatch; using non-EMA model weights")
        return False

    for p, s in zip(params, shadow):
        p.data.copy_(s.to(p.device))
    print("[mae-test] Applied MAE EMA weights from checkpoint")
    return True


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if all(k.startswith("model.") for k in keys):
        return {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _apply_ema_weights(model: torch.nn.Module, ema_path: Path) -> None:
    ema_state = torch.load(ema_path, map_location="cpu")
    shadow = ema_state.get("shadow_params")
    if shadow is None:
        raise ValueError(f"EMA file missing shadow_params: {ema_path}")
    params = list(model.parameters())
    if len(params) != len(shadow):
        raise ValueError("EMA shadow length does not match model parameters")
    for p, s in zip(params, shadow):
        p.data.copy_(s.to(p.device))


def _build_tokenizer(cfg: dict, device: torch.device) -> TokenizerHandle:
    model_name = cfg["model_name"]
    model_path = Path(cfg["model_path"])
    use_ema = bool(cfg.get("use_ema", False))

    if model_name == "Phaedra_AE_FSQ_4x4":
        task_class = PhaedraAEFSQSystem
        config_path = _tok_config("Phaedra_AE_FSQ_4x4")
        model_type = "phaedra"
    elif model_name == "AE_FSQ":
        task_class = FSQAESystem
        config_path = _tok_config("AE_FSQ")
        model_type = "fsq"
    else:
        raise ValueError(f"Unsupported tokenizer model_name: {model_name}")

    task = task_class(OmegaConf.load(config_path))
    from hub.utils.weights import is_released
    released = is_released(model_path)        # released safetensors already hold the EMA weights
    state_dict = _normalize_state_dict_keys(_load_state_dict_from_path(model_path))
    load_result = task.model.load_state_dict(state_dict, strict=released)
    missing = len(load_result.missing_keys)
    unexpected = len(load_result.unexpected_keys)
    loaded = len(task.model.state_dict()) - missing
    print(
        f"[mae-test] tokenizer load: loaded={loaded} missing={missing} unexpected={unexpected} "
        f"from {model_path}"
    )
    if use_ema and not released:
        ema_path = model_path / "ema.pt" if model_path.is_dir() else model_path.parent / "ema.pt"
        if not ema_path.exists():
            raise FileNotFoundError(f"EMA path not found: {ema_path}")
        _apply_ema_weights(task.model, ema_path)

    task.model.to(device)
    task.model.eval()
    return TokenizerHandle(task=task, model_type=model_type, device=device)


def _decode_tokens(
    tokenizer: TokenizerHandle,
    fsq_tokens: torch.Tensor | None,
    amp_tokens: torch.Tensor | None,
    morph_tokens: torch.Tensor | None,
) -> torch.Tensor:
    if tokenizer.model_type == "fsq":
        if fsq_tokens is None:
            raise ValueError("FSQ decode requires fsq_tokens")
        embeds = tokenizer.task.model.quantizer.get_codebook_entry(fsq_tokens)
        return tokenizer.task.model.decode(embeds)

    if amp_tokens is None or morph_tokens is None:
        raise ValueError("Phaedra decode requires amp_tokens and morph_tokens")
    morph_emb = tokenizer.task.model.quantizer.get_codebook_entry(morph_tokens)
    amp_emb = tokenizer.task.model.approximate_continuous.get_codebook_entry(amp_tokens)
    embeds = torch.cat([morph_emb, amp_emb], dim=1)
    return tokenizer.task.model.decode(embeds)


def _copy_unmasked_tokens(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    merged = target_tokens.clone()
    merged[mask] = pred_tokens[mask]
    return merged


def _token_accuracy_percent(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    pred_flat = pred_tokens.view(-1)
    target_flat = target_tokens.view(-1)
    mask_flat = mask.view(-1).bool()

    overall = (pred_flat == target_flat).float().mean().item() * 100.0
    if mask_flat.any():
        masked = (pred_flat[mask_flat] == target_flat[mask_flat]).float().mean().item() * 100.0
    else:
        masked = float("nan")
    return float(overall), float(masked)


def _relative_errors(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    l1 = np.mean(np.abs(pred - target))
    l2 = np.sqrt(np.mean((pred - target) ** 2))
    denom_l1 = np.mean(np.abs(target)) + 1e-8
    denom_l2 = np.sqrt(np.mean(target ** 2)) + 1e-8
    return float(l1 / denom_l1), float(l2 / denom_l2)


def _wasserstein_1d(pred: np.ndarray, target: np.ndarray) -> float:
    pred_flat = np.sort(pred.reshape(-1).astype(np.float64))
    target_flat = np.sort(target.reshape(-1).astype(np.float64))

    if pred_flat.size != target_flat.size:
        n = int(min(pred_flat.size, target_flat.size))
        if n <= 0:
            return float("nan")
        pred_idx = np.linspace(0, pred_flat.size - 1, n)
        target_idx = np.linspace(0, target_flat.size - 1, n)
        pred_flat = np.interp(pred_idx, np.arange(pred_flat.size), pred_flat)
        target_flat = np.interp(target_idx, np.arange(target_flat.size), target_flat)

    return float(np.mean(np.abs(pred_flat - target_flat)))


def _upsample_tokens(tokens: np.ndarray, scale_y: int, scale_x: int | None = None) -> np.ndarray:
    sx = int(scale_x) if scale_x is not None else int(scale_y)
    sy = int(scale_y)
    return np.repeat(np.repeat(tokens, sy, axis=-2), sx, axis=-1)


def _row_minmax(images: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.stack([np.ravel(img) for img in images], axis=0)
    vals = stacked[np.isfinite(stacked)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(vals.min()), float(vals.max())


def _save_variable_composite_plot(
    save_path: Path,
    token_type: str,
    variable_name: str,
    member_id: int,
    time_idx: int,
    masked_morph: np.ndarray,
    pred_morph: np.ndarray,
    gt_morph: np.ndarray,
    masked_amp: np.ndarray,
    pred_amp: np.ndarray,
    gt_amp: np.ndarray,
    ground_truth: np.ndarray,
    recon_pred_tokens: np.ndarray,
) -> None:
    morph_vmin, morph_vmax = _row_minmax([masked_morph, pred_morph, gt_morph])
    amp_vmin, amp_vmax = _row_minmax([masked_amp, pred_amp, gt_amp])
    field_vmin, field_vmax = _row_minmax([ground_truth, recon_pred_tokens])

    token_label = "Morphology" if token_type == "phaedra" else "Tokens"
    amp_label = "Amplitude" if token_type == "phaedra" else "Tokens"
    cmap_tok = plt.get_cmap("nipy_spectral").copy()
    cmap_amp = plt.get_cmap("viridis").copy()
    cmap_field = plt.get_cmap("turbo").copy()
    cmap_tok.set_bad("white")
    cmap_amp.set_bad("white")

    with plt.rc_context(_LATEX_STYLE_RCPARAMS):
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), constrained_layout=True)
        panels = [
            (masked_morph, f"Masked Input {token_label}", cmap_tok, morph_vmin, morph_vmax),
            (pred_morph, f"Predicted {token_label}", cmap_tok, morph_vmin, morph_vmax),
            (gt_morph, f"Ground Truth {token_label}", cmap_tok, morph_vmin, morph_vmax),
            (ground_truth, "Ground Truth Field", cmap_field, field_vmin, field_vmax),
            (masked_amp, f"Masked Input {amp_label}", cmap_amp, amp_vmin, amp_vmax),
            (pred_amp, f"Predicted {amp_label}", cmap_amp, amp_vmin, amp_vmax),
            (gt_amp, f"Ground Truth {amp_label}", cmap_amp, amp_vmin, amp_vmax),
            (recon_pred_tokens, "Predicted Reconstruction", cmap_field, field_vmin, field_vmax),
        ]

        for ax, (img, name, cmap, vmin, vmax) in zip(axes.ravel(), panels):
            im = ax.imshow(img, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        fig.suptitle(f"MAE {token_type.upper()} | member={member_id} time={time_idx} var={variable_name}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _test_member_ids(meta: dict) -> list[int]:
    n = int(meta["num_members"])
    n_val = int(meta["max_num_val_samples"])
    n_test = int(meta["max_num_test_samples"])
    test_start = n - n_val - n_test + n_val
    return list(range(test_start, n))


def _parse_time_indices(raw: str | None, max_timesteps: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(range(max_timesteps))

    vals: list[int] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(f"Invalid time range token: {token}")
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if end < start:
                raise ValueError(f"Invalid decreasing time range: {token}")
            vals.extend(range(start, end + 1))
        else:
            vals.append(int(token))

    vals = list(dict.fromkeys(vals))
    bad = [v for v in vals if v < 0 or v >= max_timesteps]
    if bad:
        raise ValueError(f"Invalid time indices {bad}; valid range is [0, {max_timesteps - 1}]")
    return vals


def _resolve_dataset_indexed_value(raw_value: object, dataset_id: int, field_name: str) -> object:
    if isinstance(raw_value, list):
        if not raw_value:
            raise ValueError(f"{field_name} list must not be empty")
        if not (0 <= dataset_id < len(raw_value)):
            raise ValueError(
                f"{field_name} list supports dataset_id in [0, {len(raw_value) - 1}], got {dataset_id}"
            )
        return raw_value[dataset_id]
    return raw_value


def _resolve_eval_dataset_config(cfg: dict, dataset_id: int) -> tuple[dict, Path, Path]:
    data_cfg_raw = cfg.get("data_configs", cfg.get("data_config"))
    if data_cfg_raw is None:
        raise ValueError("Missing data_config or data_configs in test config")
    data_cfg_path = Path(str(_resolve_dataset_indexed_value(data_cfg_raw, dataset_id, "data_configs")))
    data_cfg = _load_yaml(data_cfg_path)["dataset"]

    source_raw = cfg.get("source_datasets", cfg.get("source_dataset"))
    if source_raw is None:
        source_dataset = Path(data_cfg["path"]) / f"{data_cfg['name']}.nc"
    else:
        source_dataset = Path(str(_resolve_dataset_indexed_value(source_raw, dataset_id, "source_datasets")))

    return data_cfg, data_cfg_path, source_dataset


def _build_denorm_stats_from_data_config(data_cfg: dict, variables: list[str]) -> dict[str, tuple[float, float]]:
    normalization = data_cfg.get("normalization", {}) or {}
    stats: dict[str, tuple[float, float]] = {}
    for var in variables:
        item = normalization.get(var, {}) or {}
        mean = float(item.get("mean", 0.0))
        std = float(item.get("std", 1.0))
        if std == 0.0:
            std = 1.0
        stats[var] = (mean, std)
    return stats


def _denormalize_fields(
    normalized_fields: np.ndarray,
    var_names: list[str],
    stats: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(var_names):
        mean_val, std_val = stats.get(name, (0.0, 1.0))
        out[name] = normalized_fields[i] * float(std_val) + float(mean_val)
    return out


def _load_amp_stats_cache(cache_path: Path) -> tuple[float, float] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    mean = payload.get("mean")
    std = payload.get("std")
    if mean is None or std is None:
        return None
    std = float(std)
    if std == 0.0:
        std = 1.0
    return float(mean), std


def _estimate_amp_stats_from_token_file(
    token_path: Path,
    variables: list[str],
    num_samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    with nc.Dataset(token_path, "r") as ds:
        member_size = int(ds.dimensions["member"].size)
        time_size = int(ds.dimensions["time"].size)
        sample_count = max(1, min(num_samples, member_size * time_size * max(1, len(variables))))

        total_sum = 0.0
        total_sumsq = 0.0
        total_count = 0

        for _ in range(sample_count):
            var = variables[int(rng.integers(0, len(variables)))]
            member_idx = int(rng.integers(0, member_size))
            time_idx = int(rng.integers(0, time_size))
            arr = ds.variables[f"{var}_amp"][member_idx, time_idx, :, :].astype(np.float64)
            total_sum += float(arr.sum())
            total_sumsq += float((arr ** 2).sum())
            total_count += int(arr.size)

    if total_count == 0:
        return 0.0, 1.0

    mean = total_sum / total_count
    var = max((total_sumsq / total_count) - (mean * mean), 0.0)
    std = float(np.sqrt(var))
    if std == 0.0:
        std = 1.0
    return float(mean), std


def _checkpoint_step_from_path(path: Path) -> int | None:
    stem = path.stem
    marker = "checkpoint_step_"
    if marker in stem:
        try:
            return int(stem.split(marker)[-1])
        except ValueError:
            return None
    return None


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-test-members", type=int, default=None)
    parser.add_argument("--time-indices", default=None)
    parser.add_argument("--plot-time-indices", default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random masks (the MAE draws them from the global torch RNG); "
                             "pass -1 for the unseeded behaviour of the original evaluation runs")
    parser.add_argument("--deterministic", action="store_true",
                        help="TF32 off + deterministic kernels (bit-reproducible on a given GPU model; "
                             "needs CUBLAS_WORKSPACE_CONFIG=:4096:8)")
    args = parser.parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg = _load_yaml(Path(args.config))
    train_cfg = _load_yaml(Path(cfg["mae_train_config"]))

    variables = list(train_cfg["dataset"]["variables"])
    token_type = str(train_cfg["dataset"]["token_type"])
    train_dataset_paths = list(train_cfg["dataset"].get("paths") or [train_cfg["dataset"]["path"]])
    num_datasets = len(train_dataset_paths)
    train_dataset_names = list(train_cfg["dataset"].get("dataset_names") or [f"dataset_{i}" for i in range(num_datasets)])

    dataset_id = int(cfg.get("dataset_id", 0))
    if not (0 <= dataset_id < num_datasets):
        raise ValueError(f"dataset_id must be in [0, {num_datasets - 1}], got {dataset_id}")
    dataset_id_tensor = torch.tensor([dataset_id], dtype=torch.long)

    data_cfg, data_cfg_path, source_dataset = _resolve_eval_dataset_config(cfg, dataset_id)
    if not source_dataset.exists():
        raise FileNotFoundError(f"source_dataset not found: {source_dataset}")
    dataset_name = (
        str(train_dataset_names[dataset_id])
        if dataset_id < len(train_dataset_names)
        else f"dataset_{dataset_id}"
    )

    denorm_stats = _build_denorm_stats_from_data_config(data_cfg, variables)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset_id_tensor = dataset_id_tensor.to(device)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    tokenizer = _build_tokenizer(cfg["tokenizer"], device)

    with nc.Dataset(Path(train_dataset_paths[0]), "r") as tok_ds:
        if token_type == "phaedra":
            vocab_amp = int(tok_ds.getncattr("amplitude_codebook_size"))
            vocab_morph = int(tok_ds.getncattr("morphology_codebook_size"))
            vocab_fsq = None
        else:
            vocab_amp = None
            vocab_morph = None
            vocab_fsq = int(tok_ds.getncattr("codebook_size"))
        grid_size = int(tok_ds.dimensions["token_x"].size)

    for token_path in train_dataset_paths[1:]:
        with nc.Dataset(Path(token_path), "r") as tok_ds:
            ds_grid = int(tok_ds.dimensions["token_x"].size)
            if ds_grid != grid_size:
                raise ValueError(f"token_x mismatch across train datasets: expected {grid_size}, got {ds_grid} at {token_path}")
            if token_type == "phaedra":
                ds_vocab_amp = int(tok_ds.getncattr("amplitude_codebook_size"))
                ds_vocab_morph = int(tok_ds.getncattr("morphology_codebook_size"))
                if ds_vocab_amp != vocab_amp or ds_vocab_morph != vocab_morph:
                    raise ValueError(
                        "Phaedra vocab mismatch across train datasets: "
                        f"expected amp={vocab_amp}, morph={vocab_morph}; got amp={ds_vocab_amp}, morph={ds_vocab_morph} at {token_path}"
                    )
            else:
                ds_vocab_fsq = int(tok_ds.getncattr("codebook_size"))
                if ds_vocab_fsq != vocab_fsq:
                    raise ValueError(f"FSQ vocab mismatch across train datasets: expected {vocab_fsq}, got {ds_vocab_fsq} at {token_path}")

    model_cfg = MAEConfig(
        token_type=token_type,
        vocab_amp=vocab_amp,
        vocab_morph=vocab_morph,
        vocab_fsq=vocab_fsq,
        num_vars=len(variables),
        grid_size=grid_size,
        embed_dim=int(train_cfg["model"]["embed_dim"]),
        encoder_depth=int(train_cfg["model"]["encoder_depth"]),
        decoder_depth=int(train_cfg["model"]["decoder_depth"]),
        num_heads=int(train_cfg["model"]["num_heads"]),
        mlp_ratio=int(train_cfg["model"]["mlp_ratio"]),
        mask_ratio=float(train_cfg["training"]["mask_ratio"]),
        fusion=str(train_cfg["model"]["fusion"]),
        num_datasets=num_datasets,
    )
    model = TokenMAE(model_cfg).to(device)
    checkpoint_path = Path(cfg["checkpoint"])
    checkpoint_step = _checkpoint_step_from_path(checkpoint_path)
    state_dict = _normalize_state_dict_keys(_load_state_dict_from_path(checkpoint_path))
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = len(load_result.missing_keys)
    unexpected = len(load_result.unexpected_keys)
    loaded = len(model.state_dict()) - missing
    print(
        f"[mae-test] model load: loaded={loaded} missing={missing} unexpected={unexpected} "
        f"from {checkpoint_path}"
    )
    if checkpoint_step is not None and checkpoint_step < 10000:
        print(f"[mae-test] warning: evaluating very early checkpoint step={checkpoint_step}")
    mae_ema_applied = _maybe_apply_model_ema_from_checkpoint(model, checkpoint_path, bool(cfg.get("mae_use_ema", True)))
    model.eval()

    amp_mean = None
    amp_std = None
    if token_type == "phaedra":
        amp_samples = int(train_cfg["training"].get("amp_stats_samples", 100))
        amp_seed = int(train_cfg["training"].get("amp_stats_seed", 0))
        cache_path_str = train_cfg["training"].get("amp_stats_cache")
        if cache_path_str:
            cache_path = Path(cache_path_str)
        else:
            cache_path = Path(train_cfg["validation"]["output_dir"]) / "amp_stats.json"

        cached_stats = _load_amp_stats_cache(cache_path)
        if cached_stats is not None:
            amp_mean, amp_std = cached_stats
            print(f"[mae-test] loaded amp stats from cache: mean={amp_mean:.3f} std={amp_std:.3f}")
        else:
            amp_mean, amp_std = _estimate_amp_stats_from_token_file(
                Path(train_dataset_paths[dataset_id]),
                variables,
                amp_samples,
                amp_seed,
            )
            print(f"[mae-test] estimated amp stats: mean={amp_mean:.3f} std={amp_std:.3f}")

    member_ids = _test_member_ids(data_cfg)
    max_members = args.max_test_members if args.max_test_members is not None else cfg.get("max_test_members")
    if max_members is not None:
        member_ids = member_ids[: int(max_members)]

    time_indices = _parse_time_indices(args.time_indices or cfg.get("time_indices"), int(data_cfg["num_timesteps"]))
    plot_time_indices = _parse_time_indices(
        args.plot_time_indices or cfg.get("plot_time_indices"),
        int(data_cfg["num_timesteps"]),
    )
    plot_time_index_set = set(plot_time_indices)
    max_plots = int(cfg.get("max_plots", 4))

    metrics = {
        "samples": 0,
        "variables": variables,
        "rel_l1_pred_vs_true_tokens": {v: [] for v in variables},
        "rel_l2_pred_vs_true_tokens": {v: [] for v in variables},
        "rel_l1_true_tokens_vs_true_fields": {v: [] for v in variables},
        "rel_l2_true_tokens_vs_true_fields": {v: [] for v in variables},
        "rel_l1_pred_vs_true_fields": {v: [] for v in variables},
        "rel_l2_pred_vs_true_fields": {v: [] for v in variables},
        "w1_pred_vs_true_fields": {v: [] for v in variables},
        "token_acc_overall_pct": [],
        "token_acc_masked_pct": [],
        "token_acc_amp_overall_pct": [],
        "token_acc_amp_masked_pct": [],
        "token_acc_morph_overall_pct": [],
        "token_acc_morph_masked_pct": [],
        "encode_time_s": [],
        "process_time_s": [],
        "decode_time_s": [],
    }

    n_plots = 0
    with nc.Dataset(source_dataset, "r") as src:
        for member_id in member_ids:
            print(f"Member ID: {member_id}/{member_ids[-1]}")
            for time_idx in time_indices:
                field_stack = []
                means = []
                stds = []
                for var in variables:
                    frame = src.variables[var][member_id, time_idx, :, :]
                    field_stack.append(frame)
                    mean_val, std_val = denorm_stats.get(var, (0.0, 1.0))
                    means.append(float(mean_val))
                    stds.append(float(std_val))

                field_np = np.stack(field_stack, axis=0).astype(np.float32)
                means_np = np.asarray(means, dtype=np.float32)[:, None, None]
                stds_np = np.asarray(stds, dtype=np.float32)[:, None, None]
                field_norm = (field_np - means_np) / (stds_np + 1e-6)

                with torch.inference_mode():
                    input_tensor = torch.from_numpy(field_norm).unsqueeze(1).to(device)

                    _sync_if_cuda(device)
                    encode_start = time.perf_counter()
                    tokens = tokenizer.task.produce_tokens({"field_variables_in": input_tensor})
                    _sync_if_cuda(device)
                    metrics["encode_time_s"].append(time.perf_counter() - encode_start)

                    if token_type == "phaedra":
                        morph, amp = tokens
                        true_amp = amp.unsqueeze(0).to(torch.long)
                        true_morph = morph.unsqueeze(0).to(torch.long)

                        if amp_mean is None or amp_std is None:
                            raise RuntimeError("amp_mean/std not initialized for Phaedra evaluation")

                        _sync_if_cuda(device)
                        process_start = time.perf_counter()
                        amp_pred_norm, logits_morph, mask = model(
                            tokens_amp=true_amp,
                            tokens_morph=true_morph,
                            dataset_ids=dataset_id_tensor,
                        )
                        _sync_if_cuda(device)
                        metrics["process_time_s"].append(time.perf_counter() - process_start)
                        pred_amp_raw = (amp_pred_norm * amp_std + amp_mean).round().clamp(0, model.cfg.vocab_amp - 1).long()
                        pred_morph_raw = logits_morph.argmax(dim=-1)

                        acc_amp_overall, acc_amp_masked = _token_accuracy_percent(
                            pred_amp_raw,
                            true_amp.view(1, -1),
                            mask,
                        )
                        
                        acc_morph_overall, acc_morph_masked = _token_accuracy_percent(
                            pred_morph_raw,
                            true_morph.view(1, -1),
                            mask,
                        )
                        metrics["token_acc_amp_overall_pct"].append(acc_amp_overall)
                        metrics["token_acc_amp_masked_pct"].append(acc_amp_masked)
                        metrics["token_acc_morph_overall_pct"].append(acc_morph_overall)
                        metrics["token_acc_morph_masked_pct"].append(acc_morph_masked)
                        metrics["token_acc_overall_pct"].append(float((acc_amp_overall + acc_morph_overall) / 2.0))
                        metrics["token_acc_masked_pct"].append(float((acc_amp_masked + acc_morph_masked) / 2.0))

                        # Merge unmasked ground-truth tokens into predictions before decoding.
                        pred_amp = _copy_unmasked_tokens(pred_amp_raw, true_amp.view(1, -1), mask).view_as(true_amp)
                        pred_morph = _copy_unmasked_tokens(pred_morph_raw, true_morph.view(1, -1), mask).view_as(true_morph)
                    else:
                        true_tokens = tokens.unsqueeze(0).to(torch.long)

                        _sync_if_cuda(device)
                        process_start = time.perf_counter()
                        pred_tokens_raw, mask = model.predict(tokens_fsq=true_tokens, dataset_ids=dataset_id_tensor)
                        _sync_if_cuda(device)
                        metrics["process_time_s"].append(time.perf_counter() - process_start)
                        acc_overall, acc_masked = _token_accuracy_percent(
                            pred_tokens_raw,
                            true_tokens.view(1, -1),
                            mask,
                        )
                        metrics["token_acc_overall_pct"].append(acc_overall)
                        metrics["token_acc_masked_pct"].append(acc_masked)

                        # Merge unmasked ground-truth tokens into predictions before decoding.
                        pred_tokens = _copy_unmasked_tokens(pred_tokens_raw, true_tokens.view(1, -1), mask).view_as(true_tokens)

                    mask_grid = mask.view(1, len(variables), grid_size, grid_size)

                sample_decode_time = 0.0
                for v_idx, var in enumerate(variables):
                    _sync_if_cuda(device)
                    decode_start = time.perf_counter()
                    with torch.inference_mode():
                        if token_type == "phaedra":
                            recon_true = _decode_tokens(
                                tokenizer,
                                fsq_tokens=None,
                                amp_tokens=true_amp[0, v_idx].unsqueeze(0).to(device),
                                morph_tokens=true_morph[0, v_idx].unsqueeze(0).to(device),
                            )
                            recon_pred = _decode_tokens(
                                tokenizer,
                                fsq_tokens=None,
                                amp_tokens=pred_amp[0, v_idx].unsqueeze(0).to(device),
                                morph_tokens=pred_morph[0, v_idx].unsqueeze(0).to(device),
                            )
                        else:
                            recon_true = _decode_tokens(
                                tokenizer,
                                fsq_tokens=true_tokens[0, v_idx].unsqueeze(0).to(device),
                                amp_tokens=None,
                                morph_tokens=None,
                            )
                            recon_pred = _decode_tokens(
                                tokenizer,
                                fsq_tokens=pred_tokens[0, v_idx].unsqueeze(0).to(device),
                                amp_tokens=None,
                                morph_tokens=None,
                            )
                    _sync_if_cuda(device)
                    sample_decode_time += time.perf_counter() - decode_start

                    var_mean, var_std = denorm_stats.get(var, (0.0, 1.0))

                    recon_true_np = recon_true[0, 0].detach().cpu().numpy().astype(np.float32)
                    recon_pred_np = recon_pred[0, 0].detach().cpu().numpy().astype(np.float32)
                    recon_true_np = recon_true_np * var_std + var_mean
                    recon_pred_np = recon_pred_np * var_std + var_mean
                    gt_field_np = field_np[v_idx].astype(np.float32)

                    l1_tok, l2_tok = _relative_errors(recon_pred_np, recon_true_np)
                    l1_true_tok_field, l2_true_tok_field = _relative_errors(recon_true_np, gt_field_np)
                    l1_field, l2_field = _relative_errors(recon_pred_np, gt_field_np)
                    w1_field = _wasserstein_1d(recon_pred_np, gt_field_np)

                    metrics["rel_l1_pred_vs_true_tokens"][var].append(l1_tok)
                    metrics["rel_l2_pred_vs_true_tokens"][var].append(l2_tok)
                    metrics["rel_l1_true_tokens_vs_true_fields"][var].append(l1_true_tok_field)
                    metrics["rel_l2_true_tokens_vs_true_fields"][var].append(l2_true_tok_field)
                    metrics["rel_l1_pred_vs_true_fields"][var].append(l1_field)
                    metrics["rel_l2_pred_vs_true_fields"][var].append(l2_field)
                    metrics["w1_pred_vs_true_fields"][var].append(w1_field)

                    if n_plots < max_plots and time_idx in plot_time_index_set:
                        mask_np = mask_grid[0, v_idx].detach().cpu().numpy().astype(bool)

                        if token_type == "phaedra":
                            gt_morph_tok = true_morph[0, v_idx].detach().cpu().numpy().astype(np.float32)
                            pred_morph_tok = pred_morph[0, v_idx].detach().cpu().numpy().astype(np.float32)
                            gt_amp_tok = true_amp[0, v_idx].detach().cpu().numpy().astype(np.float32)
                            pred_amp_tok = pred_amp[0, v_idx].detach().cpu().numpy().astype(np.float32)
                        else:
                            gt_tok = true_tokens[0, v_idx].detach().cpu().numpy().astype(np.float32)
                            pred_tok = pred_tokens[0, v_idx].detach().cpu().numpy().astype(np.float32)
                            gt_morph_tok = gt_tok
                            pred_morph_tok = pred_tok
                            gt_amp_tok = gt_tok
                            pred_amp_tok = pred_tok

                        masked_morph_tok = gt_morph_tok.copy().astype(np.float32)
                        masked_amp_tok = gt_amp_tok.copy().astype(np.float32)
                        masked_morph_tok[mask_np] = np.nan
                        masked_amp_tok[mask_np] = np.nan

                        tok_h, tok_w = int(gt_morph_tok.shape[-2]), int(gt_morph_tok.shape[-1])
                        scale_y = max(1, int(gt_field_np.shape[-2]) // max(1, tok_h))
                        scale_x = max(1, int(gt_field_np.shape[-1]) // max(1, tok_w))

                        plot_path = output_dir / "plots" / f"member_{member_id}_time_{time_idx}_{var}_composite.png"
                        _save_variable_composite_plot(
                            save_path=plot_path,
                            token_type=token_type,
                            variable_name=var,
                            member_id=int(member_id),
                            time_idx=int(time_idx),
                            masked_morph=_upsample_tokens(masked_morph_tok, scale_y, scale_x),
                            pred_morph=_upsample_tokens(pred_morph_tok, scale_y, scale_x),
                            gt_morph=_upsample_tokens(gt_morph_tok, scale_y, scale_x),
                            masked_amp=_upsample_tokens(masked_amp_tok, scale_y, scale_x),
                            pred_amp=_upsample_tokens(pred_amp_tok, scale_y, scale_x),
                            gt_amp=_upsample_tokens(gt_amp_tok, scale_y, scale_x),
                            ground_truth=gt_field_np,
                            recon_pred_tokens=recon_pred_np,
                        )
                        n_plots += 1

                metrics["decode_time_s"].append(sample_decode_time)
                metrics["samples"] += 1

    l1_phys_vals = [
        float(val)
        for vals in metrics["rel_l1_pred_vs_true_fields"].values()
        for val in vals
        if np.isfinite(val)
    ]
    w1_phys_vals = [
        float(val)
        for vals in metrics["w1_pred_vs_true_fields"].values()
        for val in vals
        if np.isfinite(val)
    ]

    summary = {
        "samples": metrics["samples"],
        "token_type": token_type,
        "checkpoint": str(cfg["checkpoint"]),
        "checkpoint_step": checkpoint_step,
        "data_config": str(data_cfg_path),
        "dataset_name": dataset_name,
        "source_dataset": str(source_dataset),
        "max_test_members": max_members,
        "time_indices": time_indices,
        "plot_time_indices": plot_time_indices,
        "dataset_id": dataset_id,
        "mae_ema_applied": bool(mae_ema_applied),
        "physical_space_errors": {
            "relative_l1_pred_vs_true_fields_avg": float(np.mean(l1_phys_vals)) if l1_phys_vals else float("nan"),
            "w1_pred_vs_true_fields_avg": float(np.mean(w1_phys_vals)) if w1_phys_vals else float("nan"),
        },
        "token_metrics": {
            "token_acc_overall_pct": float(np.nanmean(metrics["token_acc_overall_pct"])) if metrics["token_acc_overall_pct"] else float("nan"),
            "token_acc_masked_pct": float(np.nanmean(metrics["token_acc_masked_pct"])) if metrics["token_acc_masked_pct"] else float("nan"),
        },
        "timing_seconds": {
            "encode_total_s": float(np.sum(metrics["encode_time_s"])) if metrics["encode_time_s"] else float("nan"),
            "process_total_s": float(np.sum(metrics["process_time_s"])) if metrics["process_time_s"] else float("nan"),
            "decode_total_s": float(np.sum(metrics["decode_time_s"])) if metrics["decode_time_s"] else float("nan"),
            "encode_avg_per_sample_s": float(np.mean(metrics["encode_time_s"])) if metrics["encode_time_s"] else float("nan"),
            "process_avg_per_sample_s": float(np.mean(metrics["process_time_s"])) if metrics["process_time_s"] else float("nan"),
            "decode_avg_per_sample_s": float(np.mean(metrics["decode_time_s"])) if metrics["decode_time_s"] else float("nan"),
        },
        "per_variable": {},
    }

    if token_type == "phaedra":
        summary["token_metrics"]["token_acc_amp_overall_pct"] = float(np.nanmean(metrics["token_acc_amp_overall_pct"])) if metrics["token_acc_amp_overall_pct"] else float("nan")
        summary["token_metrics"]["token_acc_amp_masked_pct"] = float(np.nanmean(metrics["token_acc_amp_masked_pct"])) if metrics["token_acc_amp_masked_pct"] else float("nan")
        summary["token_metrics"]["token_acc_morph_overall_pct"] = float(np.nanmean(metrics["token_acc_morph_overall_pct"])) if metrics["token_acc_morph_overall_pct"] else float("nan")
        summary["token_metrics"]["token_acc_morph_masked_pct"] = float(np.nanmean(metrics["token_acc_morph_masked_pct"])) if metrics["token_acc_morph_masked_pct"] else float("nan")

    for var in variables:
        summary["per_variable"][var] = {
            "rel_l1_pred_vs_true_tokens": float(np.mean(metrics["rel_l1_pred_vs_true_tokens"][var])) if metrics["rel_l1_pred_vs_true_tokens"][var] else float("nan"),
            "rel_l2_pred_vs_true_tokens": float(np.mean(metrics["rel_l2_pred_vs_true_tokens"][var])) if metrics["rel_l2_pred_vs_true_tokens"][var] else float("nan"),
            "rel_l1_true_tokens_vs_true_fields": float(np.mean(metrics["rel_l1_true_tokens_vs_true_fields"][var])) if metrics["rel_l1_true_tokens_vs_true_fields"][var] else float("nan"),
            "rel_l2_true_tokens_vs_true_fields": float(np.mean(metrics["rel_l2_true_tokens_vs_true_fields"][var])) if metrics["rel_l2_true_tokens_vs_true_fields"][var] else float("nan"),
            "rel_l1_pred_vs_true_fields": float(np.mean(metrics["rel_l1_pred_vs_true_fields"][var])) if metrics["rel_l1_pred_vs_true_fields"][var] else float("nan"),
            "rel_l2_pred_vs_true_fields": float(np.mean(metrics["rel_l2_pred_vs_true_fields"][var])) if metrics["rel_l2_pred_vs_true_fields"][var] else float("nan"),
            "w1_pred_vs_true_fields": float(np.mean(metrics["w1_pred_vs_true_fields"][var])) if metrics["w1_pred_vs_true_fields"][var] else float("nan"),
        }

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
