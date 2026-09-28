from __future__ import annotations
print("Beginning Imports")
import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import math
import hashlib
import shutil
import time
import json
from contextlib import nullcontext

import netCDF4 as nc
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mae.src.dataset import TokenDataset, TokenDatasetConfig
from mae.src.model import MAEConfig, TokenMAE
print("Finished Importing")

REPO_ROOT = Path(__file__).resolve().parents[1]
from tokenizer import config_path as _tok_config  # noqa: E402


@dataclass
class DecoderHandle:
    task: object
    model_type: str
    device: torch.device


def _load_config(path: Path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _configure_sdp(mode: str) -> None:
    if not torch.cuda.is_available():
        return

    mode = mode.lower()
    if mode == "auto":
        return

    enable_flash = mode == "flash"
    enable_mem = mode == "mem_efficient"
    enable_math = mode == "math"

    try:
        torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem,
        )
    except Exception:
        torch.backends.cuda.enable_flash_sdp(enable_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem)
        torch.backends.cuda.enable_math_sdp(enable_math)


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    return dist.get_rank() if _dist_is_initialized() else 0


def _is_main_process() -> bool:
    return _dist_rank() == 0


def _dist_barrier() -> None:
    if _dist_is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def _broadcast_object(obj: object, src: int = 0) -> object:
    if not _dist_is_initialized():
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _resolve_tmp_root(tmp_override: str | None) -> Path:
    if tmp_override:
        return Path(os.path.expanduser(tmp_override))
    for key in ("TMPDIR", "SLURM_TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    return Path("/tmp")


def _acquire_lock(lock_path: Path):
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        return fd
    except FileExistsError:
        return None


def _wait_for_copy(lock_path: Path, expected_path: Path, timeout_s: int = 3600) -> bool:
    start = time.time()
    while lock_path.exists():
        if time.time() - start > timeout_s:
            return False
        time.sleep(5)
    return expected_path.exists()


def _copy_dataset_to_tmp(src_path: Path, tmp_override: str | None) -> Path:
    src_path = Path(os.path.expanduser(str(src_path))).resolve()
    tmp_root = _resolve_tmp_root(tmp_override)
    tmp_root.mkdir(parents=True, exist_ok=True)

    suffix = hashlib.sha1(str(src_path).encode("utf-8")).hexdigest()[:8]
    dest_path = tmp_root / f"{src_path.name}_{suffix}"
    lock_path = tmp_root / f".{src_path.name}_{suffix}.lock"

    if dest_path.exists():
        return dest_path

    lock_fd = _acquire_lock(lock_path)
    if lock_fd is None:
        if _wait_for_copy(lock_path, dest_path):
            return dest_path
        return src_path

    try:
        if src_path.is_dir():
            shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest_path)
        return dest_path
    finally:
        os.close(lock_fd)
        if lock_path.exists():
            lock_path.unlink()


def _resolve_dataset_paths(dataset_section: dict) -> list[str]:
    paths_cfg = dataset_section.get("paths")
    if paths_cfg:
        return [str(Path(p).expanduser()) for p in paths_cfg]

    path_cfg = dataset_section.get("path")
    if path_cfg:
        return [str(Path(path_cfg).expanduser())]

    raise ValueError("dataset.path or dataset.paths must be provided")


def _resolve_dataset_names(dataset_section: dict, dataset_paths: list[str]) -> list[str]:
    names_cfg = dataset_section.get("dataset_names")
    if names_cfg:
        if len(names_cfg) != len(dataset_paths):
            raise ValueError(
                f"dataset_names length ({len(names_cfg)}) must match dataset paths ({len(dataset_paths)})"
            )
        return [str(name) for name in names_cfg]
    return [Path(path).stem for path in dataset_paths]


def _dataset_cache_key(dataset_paths: list[str]) -> str:
    return "|".join(dataset_paths)


def _relative_errors(pred: np.ndarray, target: np.ndarray):
    l1 = np.mean(np.abs(pred - target))
    l2 = np.sqrt(np.mean((pred - target) ** 2))
    denom_l1 = np.mean(np.abs(target)) + 1e-8
    denom_l2 = np.sqrt(np.mean(target ** 2)) + 1e-8
    return l1 / denom_l1, l2 / denom_l2


def _upsample_tokens(tokens: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(tokens, scale, axis=-2), scale, axis=-1)


def _row_minmax(images: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.stack([np.ravel(img) for img in images], axis=0)
    vals = stacked[np.isfinite(stacked)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(vals.min()), float(vals.max())


def _save_grid_png(
    path: Path,
    rows: list[list[np.ndarray]],
    titles: list[list[str]],
    row_norms: list[tuple[float, float]] | None = None,
) -> None:
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(10, 6))
    if len(rows) == 1:
        axes = np.array([axes])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    for r_idx, row in enumerate(rows):
        row_vmin, row_vmax = row_norms[r_idx] if row_norms is not None else (None, None)
        for c_idx, img in enumerate(row):
            ax = axes[r_idx, c_idx]
            ax.imshow(img, interpolation="nearest", cmap=cmap, vmin=row_vmin, vmax=row_vmax)
            ax.set_title(titles[r_idx][c_idx])
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _token_utilization(tokens: torch.Tensor, vocab_size: int, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        tokens = tokens[mask]
    if tokens.numel() == 0:
        return 0.0
    unique = torch.unique(tokens).numel()
    return float(unique) / float(vocab_size)


def _copy_unmasked_tokens(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred_tokens.shape != target_tokens.shape or pred_tokens.shape != mask.shape:
        raise ValueError(
            "pred_tokens, target_tokens, and mask must have identical shapes; "
            f"got {pred_tokens.shape}, {target_tokens.shape}, {mask.shape}"
        )
    merged = target_tokens.clone()
    merged[mask] = pred_tokens[mask]
    return merged


def _estimate_amp_stats(dataset: TokenDataset, num_samples: int = 100, seed: int = 0) -> tuple[float, float]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    total_count = 0
    mean = 0.0
    m2 = 0.0
    rng = np.random.default_rng(seed)
    sample_count = min(num_samples, len(dataset))
    indices = rng.choice(len(dataset), size=sample_count, replace=False)

    for idx in indices:
        sample = dataset[int(idx)]
        amp = sample["amp"].detach().cpu().numpy().astype(np.float64)
        values = amp.reshape(-1)
        for value in values:
            total_count += 1
            delta = value - mean
            mean += delta / total_count
            delta2 = value - mean
            m2 += delta * delta2

    if total_count < 2:
        return float(mean), 1.0

    variance = m2 / (total_count - 1)
    std = float(np.sqrt(variance))
    if std == 0.0:
        std = 1.0
    return float(mean), std


def _load_amp_stats_cache(cache_path: Path, dataset_path: str, num_samples: int, seed: int) -> tuple[float, float] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    if (
        payload.get("dataset_path") != dataset_path
        or payload.get("num_samples") != num_samples
        or payload.get("seed") != seed
    ):
        return None

    mean = float(payload.get("mean", 0.0))
    std = float(payload.get("std", 0.0))
    if std == 0.0:
        std = 1.0
    return mean, std


def _save_amp_stats_cache(cache_path: Path, dataset_path: str, num_samples: int, seed: int, mean: float, std: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": dataset_path,
        "num_samples": num_samples,
        "seed": seed,
        "mean": mean,
        "std": std,
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _assert_token_range(name: str, tokens: torch.Tensor, vocab_size: int) -> None:
    if tokens.dtype != torch.long:
        raise ValueError(f"{name} tokens must be int64; got {tokens.dtype}")
    if tokens.numel() == 0:
        return
    tok_min = int(tokens.min().item())
    tok_max = int(tokens.max().item())
    if tok_min < 0 or tok_max >= vocab_size:
        raise ValueError(f"{name} tokens out of range [0, {vocab_size - 1}]: min={tok_min} max={tok_max}")


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if all(k.startswith("model.") for k in keys):
        return {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_state_dict_from_path(path: Path) -> dict:
    """Weights from a released `model.safetensors`, a training checkpoint, or a directory."""
    from hub.utils.weights import load_weights
    return load_weights(path)[0]


def _apply_ema_weights(model: torch.nn.Module, ema_state: dict) -> None:
    shadow = ema_state.get("shadow_params")
    if shadow is None:
        raise ValueError("ema.pt does not contain shadow_params")
    params = list(model.parameters())
    if len(shadow) != len(params):
        raise ValueError("EMA parameter count does not match model parameters")
    for p, s in zip(params, shadow):
        p.data.copy_(s.to(p.device))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1); got {decay}")
        self.decay = decay
        self.shadow_params = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow, param in zip(self.shadow_params, model.parameters()):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: torch.nn.Module) -> None:
        for param, shadow in zip(model.parameters(), self.shadow_params):
            param.data.copy_(shadow)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": [p.detach().cpu() for p in self.shadow_params],
        }

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.decay = float(state["decay"])
        self.shadow_params = [p.to(device) for p in state["shadow_params"]]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        if num_classes <= 1:
            raise ValueError("FocalLoss requires logits with at least 2 classes")

        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        ce = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth_loss

        with torch.no_grad():
            p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    ema: EMA | None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)


def _load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, ema: EMA | None) -> dict:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"], device)
    return state


def _load_warm_start_weights(path: Path, model: torch.nn.Module, device: torch.device) -> tuple[int, int, int, int]:
    # A released pre-training directory may ship `model_raw.safetensors` next to the
    # (EMA) evaluation weights: the paper's fine-tuning runs started from the raw weights.
    if Path(path).is_dir() and (Path(path) / "model_raw.safetensors").is_file():
        path = Path(path) / "model_raw.safetensors"
    state_dict = _load_state_dict_from_path(path)
    state_dict = _normalize_state_dict_keys(state_dict)
    model_state = model.state_dict()

    compatible = {}
    shape_mismatch = 0
    unexpected = 0

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected += 1
            continue
        if model_state[key].shape != value.shape:
            shape_mismatch += 1
            continue
        compatible[key] = value.to(device=model_state[key].device, dtype=model_state[key].dtype)

    load_result = model.load_state_dict(compatible, strict=False)
    loaded = len(compatible)
    missing = len(load_result.missing_keys)
    return loaded, missing, unexpected, shape_mismatch


def _load_decoder(cfg: dict, device: torch.device) -> DecoderHandle | None:
    decoder_cfg = cfg.get("decoder")
    if not decoder_cfg or not decoder_cfg.get("enabled", False):
        return None

    from tokenizer.systems import PhaedraAEFSQSystem, FSQAESystem

    model_name = decoder_cfg["model_name"]
    model_path = Path(decoder_cfg["model_path"])
    use_ema = bool(decoder_cfg.get("use_ema", False))

    if model_name == "Phaedra_AE_FSQ_4x4":
        task_class = PhaedraAEFSQSystem
        config_path = _tok_config("Phaedra_AE_FSQ_4x4")
        model_type = "phaedra"
    elif model_name == "AE_FSQ":
        task_class = FSQAESystem
        config_path = _tok_config("AE_FSQ")
        model_type = "fsq"
    else:
        raise ValueError(f"Unsupported decoder model_name: {model_name}")

    model_config = OmegaConf.load(config_path)
    task = task_class(model_config)

    from hub.utils.weights import is_released
    released = is_released(model_path)        # released safetensors already hold the EMA weights
    state_dict = _load_state_dict_from_path(model_path)
    state_dict = _normalize_state_dict_keys(state_dict)
    missing, unexpected = task.model.load_state_dict(state_dict, strict=released)
    if missing or unexpected:
        print(f"[decoder] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    if use_ema and not released:
        ema_path = model_path / "ema.pt" if model_path.is_dir() else model_path.parent / "ema.pt"
        if not ema_path.exists():
            raise FileNotFoundError(f"ema.pt not found at {ema_path}")
        ema_state = torch.load(ema_path, map_location="cpu")
        _apply_ema_weights(task.model, ema_state)

    task.model.to(device)
    task.model.eval()
    return DecoderHandle(task=task, model_type=model_type, device=device)


def _decode_from_tokens(decoder: DecoderHandle, amp_tokens: torch.Tensor | None, morph_tokens: torch.Tensor | None, fsq_tokens: torch.Tensor | None) -> torch.Tensor:
    if decoder.model_type == "phaedra":
        if amp_tokens is None or morph_tokens is None:
            raise ValueError("Phaedra decoder requires amp and morph tokens")
        morph_embeddings = decoder.task.model.quantizer.get_codebook_entry(morph_tokens)
        amp_embeddings = decoder.task.model.approximate_continuous.get_codebook_entry(amp_tokens)
        embeddings = torch.cat([morph_embeddings, amp_embeddings], dim=1)
        return decoder.task.model.decode(embeddings)
    if fsq_tokens is None:
        raise ValueError("FSQ decoder requires tokens")
    embeddings = decoder.task.model.quantizer.get_codebook_entry(fsq_tokens)
    return decoder.task.model.decode(embeddings)

def get_parameter_groups(model: torch.nn.Module, weight_decay: float):
    decay = []
    no_decay = []
    # Explicitly protect your custom embeddings/tokens from decay
    skip_list = {"pos_embed", "mask_token", "var_embed", "dataset_embed"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Biases and LayerNorm parameters are 1D. We also check the skip_list.
        if len(param.shape) == 1 or any(k in name for k in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

def validate(
    model: TokenMAE,
    batch,
    cfg,
    output_dir: Path,
    decoder: DecoderHandle | None,
    dataset_label: str | None = None,
    ema: EMA | None = None,
    amp_mean: float | None = None,
    amp_std: float | None = None,
):
    model.eval()
    token_type = cfg["dataset"]["token_type"]
    variables = cfg["dataset"]["variables"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}

    ema_backup = None
    if ema is not None and cfg.get("ema", {}).get("use_for_eval", False):
        ema_backup = [p.detach().clone() for p in model.parameters()]
        ema.copy_to(model)

    dataset_ids = batch["dataset_id"].to(dtype=torch.long)
    dataset_tag = dataset_label if dataset_label is not None else f"dataset_{int(dataset_ids[0].item())}"
    dataset_tag = str(dataset_tag).replace(" ", "_").replace("/", "_")

    with torch.inference_mode():
        if token_type == "phaedra":
            amp_pred, logits_morph, mask = model(
                batch["amp"],
                batch["morph"],
                None,
                dataset_ids=dataset_ids,
            )
            if amp_mean is not None and amp_std is not None:
                pred_amp = (amp_pred * amp_std + amp_mean).round().clamp(0, model.cfg.vocab_amp - 1).long()
            else:
                pred_amp = amp_pred.round().clamp(0, model.cfg.vocab_amp - 1).long()
            pred_morph = logits_morph.argmax(dim=-1)
            target_amp = batch["amp"].view(batch["amp"].shape[0], -1)
            target_morph = batch["morph"].view(batch["morph"].shape[0], -1)
            pred_amp = _copy_unmasked_tokens(pred_amp, target_amp, mask)
            pred_morph = _copy_unmasked_tokens(pred_morph, target_morph, mask)
        else:
            logits, mask = model(None, None, batch["tokens"], dataset_ids=dataset_ids)
            pred = logits.argmax(dim=-1)
            target = batch["tokens"].view(batch["tokens"].shape[0], -1)
            pred = _copy_unmasked_tokens(pred, target, mask)

    if ema_backup is not None:
        for param, backup in zip(model.parameters(), ema_backup):
            param.data.copy_(backup)

    grid_size = model.grid_size
    scale = 128 // grid_size
    member_id = int(batch["member_idx"][0].item())
    time_id = int(batch["time_idx"][0].item())
    sample_id = f"{dataset_tag}_member_{member_id}_time_{time_id}"

    if token_type == "phaedra":
        l1_amp_vals = []
        l2_amp_vals = []
        l1_m_vals = []
        l2_m_vals = []
        batch_size = target_amp.shape[0]
        pred_amp = pred_amp.view(batch_size, len(variables), grid_size, grid_size)
        pred_morph = pred_morph.view(batch_size, len(variables), grid_size, grid_size)
        target_amp = target_amp.view(batch_size, len(variables), grid_size, grid_size)
        target_morph = target_morph.view(batch_size, len(variables), grid_size, grid_size)
        mask_grid = mask.view(batch_size, len(variables), grid_size, grid_size)

        for v_idx, var in enumerate(variables):
            amp_gt = target_amp[0, v_idx].cpu().numpy()
            amp_pred = pred_amp[0, v_idx].cpu().numpy()
            morph_gt = target_morph[0, v_idx].cpu().numpy()
            morph_pred = pred_morph[0, v_idx].cpu().numpy()

            amp_gt_up = _upsample_tokens(amp_gt, scale)
            amp_pred_up = _upsample_tokens(amp_pred, scale)
            morph_gt_up = _upsample_tokens(morph_gt, scale)
            morph_pred_up = _upsample_tokens(morph_pred, scale)

            rel_l1_amp, rel_l2_amp = _relative_errors(amp_pred_up, amp_gt_up)
            rel_l1_m, rel_l2_m = _relative_errors(morph_pred_up, morph_gt_up)
            l1_amp_vals.append(rel_l1_amp)
            l2_amp_vals.append(rel_l2_amp)
            l1_m_vals.append(rel_l1_m)
            l2_m_vals.append(rel_l2_m)

            # np.save(output_dir / f"{sample_id}_{var}_amp_gt.npy", amp_gt_up)
            # np.save(output_dir / f"{sample_id}_{var}_amp_pred.npy", amp_pred_up)
            # np.save(output_dir / f"{sample_id}_{var}_morph_gt.npy", morph_gt_up)
            # np.save(output_dir / f"{sample_id}_{var}_morph_pred.npy", morph_pred_up)

            mask_np = mask_grid[0, v_idx].cpu().numpy().astype(bool)
            masked_amp = amp_gt.copy().astype(float)
            masked_morph = morph_gt.copy().astype(float)
            masked_amp[mask_np] = np.nan
            masked_morph[mask_np] = np.nan
            masked_amp_up = _upsample_tokens(masked_amp, scale)
            masked_morph_up = _upsample_tokens(masked_morph, scale)

            diff_amp = np.abs(amp_pred_up - amp_gt_up) + amp_mean if amp_mean is not None else np.abs(amp_pred_up - amp_gt_up)
            diff_morph = np.abs(morph_pred_up - morph_gt_up)
            png_path = output_dir / f"{sample_id}_{var}_tokens.png"
            amp_vmin, amp_vmax = _row_minmax([amp_gt_up, amp_pred_up])
            morph_vmin, morph_vmax = _row_minmax([morph_gt_up, morph_pred_up])
            _save_grid_png(
                png_path,
                [
                    [amp_gt_up, masked_amp_up, amp_pred_up, diff_amp],
                    [morph_gt_up, masked_morph_up, morph_pred_up, diff_morph],
                ],
                [
                    ["amp_gt", "amp_masked", "amp_pred", "amp_diff"],
                    ["morph_gt", "morph_masked", "morph_pred", "morph_diff"],
                ],
                [(amp_vmin, amp_vmax), (morph_vmin, morph_vmax)],
            )
            if v_idx == 0:
                _save_grid_png(
                    output_dir / f"{sample_id}_mae.png",
                    [
                        [amp_gt_up, masked_amp_up, amp_pred_up, diff_amp],
                        [morph_gt_up, masked_morph_up, morph_pred_up, diff_morph],
                    ],
                    [
                        ["amp_gt", "amp_masked", "amp_pred", "amp_diff"],
                        ["morph_gt", "morph_masked", "morph_pred", "morph_diff"],
                    ],
                    [(amp_vmin, amp_vmax), (morph_vmin, morph_vmax)],
                )

            if decoder is not None:
                amp_tok = target_amp[0, v_idx].unsqueeze(0).to(decoder.device)
                morph_tok = target_morph[0, v_idx].unsqueeze(0).to(decoder.device)
                amp_pred_tok = pred_amp[0, v_idx].unsqueeze(0).to(decoder.device)
                morph_pred_tok = pred_morph[0, v_idx].unsqueeze(0).to(decoder.device)
                with torch.inference_mode():
                    recon_gt = _decode_from_tokens(decoder, amp_tok, morph_tok, None)
                    recon_pred = _decode_from_tokens(decoder, amp_pred_tok, morph_pred_tok, None)
                recon_gt_np = recon_gt.squeeze().detach().cpu().numpy()
                recon_pred_np = recon_pred.squeeze().detach().cpu().numpy()
                recon_diff = recon_pred_np - recon_gt_np
                recon_path = output_dir / f"{sample_id}_{var}_recon.png"
                _save_grid_png(
                    recon_path,
                    [[recon_gt_np, recon_pred_np, recon_diff]],
                    [["recon_gt_tokens", "recon_pred_tokens", "recon_diff"]],
                )

            # print(
            #     f"Val {var} amp rel_l1={rel_l1_amp:.4f} rel_l2={rel_l2_amp:.4f} "
            #     f"morph rel_l1={rel_l1_m:.4f} rel_l2={rel_l2_m:.4f}"
            # )
        metrics["val_rel_l1_amp"] = float(np.mean(l1_amp_vals))
        metrics["val_rel_l2_amp"] = float(np.mean(l2_amp_vals))
        metrics["val_rel_l1_morph"] = float(np.mean(l1_m_vals))
        metrics["val_rel_l2_morph"] = float(np.mean(l2_m_vals))
        metrics["val_util_pred_amp"] = _token_utilization(pred_amp, model.cfg.vocab_amp)
        metrics["val_util_pred_morph"] = _token_utilization(pred_morph, model.cfg.vocab_morph)
        metrics["val_util_target_amp"] = _token_utilization(target_amp, model.cfg.vocab_amp)
        metrics["val_util_target_morph"] = _token_utilization(target_morph, model.cfg.vocab_morph)
    else:
        l1_vals = []
        l2_vals = []
        batch_size = target.shape[0]
        pred = pred.view(batch_size, len(variables), grid_size, grid_size)
        target = target.view(batch_size, len(variables), grid_size, grid_size)
        mask_grid = mask.view(batch_size, len(variables), grid_size, grid_size)
        for v_idx, var in enumerate(variables):
            gt = target[0, v_idx].cpu().numpy()
            pred_grid = pred[0, v_idx].cpu().numpy()
            mask_np = mask_grid[0, v_idx].cpu().numpy().astype(bool)
            masked_gt = gt.copy().astype(float)
            masked_gt[mask_np] = np.nan

            gt_up = _upsample_tokens(gt, scale)
            pred_up = _upsample_tokens(pred_grid, scale)

            rel_l1, rel_l2 = _relative_errors(pred_up, gt_up)
            l1_vals.append(rel_l1)
            l2_vals.append(rel_l2)
            # np.save(output_dir / f"{sample_id}_{var}_gt.npy", gt_up)
            # np.save(output_dir / f"{sample_id}_{var}_pred.npy", pred_up)

            masked_up = _upsample_tokens(masked_gt, scale)
            diff = np.abs(pred_up - gt_up)
            png_path = output_dir / f"{sample_id}_{var}_tokens.png"
            row_vmin, row_vmax = _row_minmax([gt_up, pred_up])
            _save_grid_png(
                png_path,
                [[gt_up, masked_up, pred_up, diff]],
                [["gt", "masked", "pred", "diff"]],
                [(row_vmin, row_vmax)],
            )
            if v_idx == 0:
                _save_grid_png(
                    output_dir / f"{sample_id}_mae.png",
                    [[gt_up, masked_up, pred_up, diff]],
                    [["gt", "masked", "pred", "diff"]],
                    [(row_vmin, row_vmax)],
                )

            if decoder is not None:
                tok = target[0, v_idx].unsqueeze(0).to(decoder.device)
                tok_pred = pred[0, v_idx].unsqueeze(0).to(decoder.device)
                with torch.inference_mode():
                    recon_gt = _decode_from_tokens(decoder, None, None, tok)
                    recon_pred = _decode_from_tokens(decoder, None, None, tok_pred)
                recon_gt_np = recon_gt.squeeze().detach().cpu().numpy()
                recon_pred_np = recon_pred.squeeze().detach().cpu().numpy()
                recon_diff = recon_pred_np - recon_gt_np
                recon_path = output_dir / f"{sample_id}_{var}_recon.png"
                _save_grid_png(
                    recon_path,
                    [[recon_gt_np, recon_pred_np, recon_diff]],
                    [["recon_gt_tokens", "recon_pred_tokens", "recon_diff"]],
                )

            # print(f"Val {var} rel_l1={rel_l1:.4f} rel_l2={rel_l2:.4f}")
        metrics["val_rel_l1"] = float(np.mean(l1_vals))
        metrics["val_rel_l2"] = float(np.mean(l2_vals))
        metrics["val_util_pred"] = _token_utilization(pred, model.cfg.vocab_fsq)
        metrics["val_util_target"] = _token_utilization(target, model.cfg.vocab_fsq)
    return metrics


def _batch_sample(sample: dict, device: torch.device) -> dict:
    batched = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batched[k] = v.unsqueeze(0).to(device)
        else:
            batched[k] = torch.tensor([v], device=device)
    return batched


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl", device_id=local_rank)
            except TypeError:
                dist.init_process_group(backend="nccl")

    output_dir = Path(cfg["validation"]["output_dir"])
    if _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=OmegaConf.create(cfg), f=output_dir / "train_config.yaml")
    _dist_barrier()

    dataset_section = cfg["dataset"]
    dataset_paths = _resolve_dataset_paths(dataset_section)
    dataset_names = _resolve_dataset_names(dataset_section, dataset_paths)
    if _is_main_process():
        if dataset_section.get("copy_to_tmp", False):
            print(f"Copying Data to {dataset_section.get('tmp_dir')}")
        else:
            print("Using dataset in-place without copying")

    copied_dataset_paths = list(dataset_paths)
    if dataset_section.get("copy_to_tmp", False):
        if _is_main_process():
            tmp_dir = dataset_section.get("tmp_dir")
            copied_dataset_paths = [
                str(_copy_dataset_to_tmp(Path(path), tmp_dir))
                for path in dataset_paths
            ]
        _dist_barrier()

    copied_dataset_paths = _broadcast_object(copied_dataset_paths)
    dataset_paths = [str(path) for path in copied_dataset_paths]

    dataset_section["path"] = dataset_paths[0]
    dataset_section["paths"] = dataset_paths if len(dataset_paths) > 1 else None
    dataset_section["dataset_names"] = dataset_names
    cfg["dataset"]["path"] = dataset_section["path"]
    cfg["dataset"]["paths"] = dataset_section["paths"]
    cfg["dataset"]["dataset_names"] = dataset_names
    dataset_path_key = _dataset_cache_key(dataset_paths)

    if _is_main_process():
        print(f"Loading Data ({len(dataset_paths)} dataset(s)): {dataset_paths}")
    dataset_cfg = TokenDatasetConfig(
        token_type=dataset_section["token_type"],
        path=dataset_section["path"],
        paths=dataset_section.get("paths"),
        dataset_names=dataset_names,
        variables=dataset_section["variables"],
        split=dataset_section["split"],
    )
    train_dataset = TokenDataset(dataset_cfg)

    if _is_main_process():
        print(f"Preparing Dataloader. Batch size: {cfg['training']['batch_size']} | Num workers: {cfg['dataset']['num_workers']}")
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if _dist_is_initialized() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=True,
    )

    val_cfg = TokenDatasetConfig(
        token_type=cfg["dataset"]["token_type"],
        path=dataset_section["path"],
        paths=dataset_section.get("paths"),
        dataset_names=dataset_names,
        variables=cfg["dataset"]["variables"],
        split=cfg["validation"]["split"],
    )
    val_dataset = TokenDataset(val_cfg)

    if _is_main_process():
        print("Computing Statistics for Amplitude Tokens.")
    amp_mean = None
    amp_std = None
    if cfg["dataset"]["token_type"] == "phaedra":
        amp_samples = int(cfg["training"].get("amp_stats_samples", 100))
        amp_seed = int(cfg["training"].get("amp_stats_seed", 0))
        cache_path = cfg["training"].get("amp_stats_cache")
        if cache_path:
            cache_path = Path(cache_path)
        else:
            cache_path = output_dir / "amp_stats.json"

        if _is_main_process():
            cached = _load_amp_stats_cache(cache_path, dataset_path_key, amp_samples, amp_seed)
            if cached is not None:
                amp_mean, amp_std = cached
                print(f"[amp-stats] cached mean={amp_mean:.3f} std={amp_std:.3f} ({cache_path})")
            else:
                amp_mean, amp_std = _estimate_amp_stats(train_dataset, amp_samples, amp_seed)
                _save_amp_stats_cache(cache_path, dataset_path_key, amp_samples, amp_seed, amp_mean, amp_std)
                print(f"[amp-stats] mean={amp_mean:.3f} std={amp_std:.3f} (samples={amp_samples})")
        amp_stats = _broadcast_object({"mean": amp_mean, "std": amp_std})
        amp_mean = float(amp_stats["mean"])
        amp_std = float(amp_stats["std"])

    vocab_amp = None
    vocab_morph = None
    vocab_fsq = None
    grid_size = None
    for token_path in dataset_paths:
        with nc.Dataset(token_path, "r") as ds:
            ds_grid = int(ds.dimensions["token_x"].size)
            if cfg["dataset"]["token_type"] == "phaedra":
                ds_vocab_amp = int(ds.getncattr("amplitude_codebook_size"))
                ds_vocab_morph = int(ds.getncattr("morphology_codebook_size"))
            else:
                ds_vocab_amp = None
                ds_vocab_morph = None
                ds_vocab_fsq = int(ds.getncattr("codebook_size"))

            if grid_size is None:
                grid_size = ds_grid
                vocab_amp = ds_vocab_amp
                vocab_morph = ds_vocab_morph
                if cfg["dataset"]["token_type"] != "phaedra":
                    vocab_fsq = ds_vocab_fsq
            else:
                if ds_grid != grid_size:
                    raise ValueError(f"token_x mismatch across datasets: expected {grid_size}, got {ds_grid} at {token_path}")
                if cfg["dataset"]["token_type"] == "phaedra":
                    if ds_vocab_amp != vocab_amp or ds_vocab_morph != vocab_morph:
                        raise ValueError(
                            "Phaedra vocab mismatch across datasets: "
                            f"expected amp={vocab_amp}, morph={vocab_morph}; got amp={ds_vocab_amp}, morph={ds_vocab_morph} at {token_path}"
                        )
                else:
                    if ds_vocab_fsq != vocab_fsq:
                        raise ValueError(
                            f"FSQ vocab mismatch across datasets: expected {vocab_fsq}, got {ds_vocab_fsq} at {token_path}"
                        )

    if os.getenv("MAE_DISABLE_FLASH_SDP") == "1":
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        except Exception:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    model_cfg = MAEConfig(
        token_type=cfg["dataset"]["token_type"],
        vocab_amp=vocab_amp,
        vocab_morph=vocab_morph,
        vocab_fsq=vocab_fsq,
        num_vars=len(cfg["dataset"]["variables"]),
        grid_size=grid_size,
        embed_dim=cfg["model"]["embed_dim"],
        encoder_depth=cfg["model"]["encoder_depth"],
        decoder_depth=cfg["model"]["decoder_depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        mask_ratio=cfg["training"]["mask_ratio"],
        fusion=cfg["model"]["fusion"],
        num_datasets=len(dataset_paths),
    )

    if _is_main_process():
        print("Initializing Model.")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    mixed_precision = str(cfg["training"].get("mixed_precision", "bf16"))
    sdp_mode = str(cfg["training"].get("sdp_mode", "auto"))
    tf32_enabled = bool(cfg["training"].get("tf32", True))

    if device.type == "cuda" and tf32_enabled:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _configure_sdp(sdp_mode)

    use_amp = device.type == "cuda" and mixed_precision in {"bf16", "fp16"}
    if mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
        scaler = None
    elif mixed_precision == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = None
        scaler = None
    base_model = TokenMAE(model_cfg).to(device)
    model = base_model
    if _dist_is_initialized():
        ddp_find_unused = cfg["dataset"]["token_type"] == "fsq"
        model = torch.nn.parallel.DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            find_unused_parameters=ddp_find_unused,
        )

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if _is_main_process():
        print(f"Model params: {total_params:,} total | {trainable_params:,} trainable")

    decoder = None
    if _is_main_process():
        print("Loading Phaedra Decoder.")
        decoder = _load_decoder(cfg, device)

    param_groups = get_parameter_groups(base_model, cfg["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg["training"]["lr"],
        # We don't need weight_decay here since it's defined in the param_groups
    )

    ema = None
    ema_cfg = cfg.get("ema", {})
    if ema_cfg.get("enabled", False):
        ema = EMA(base_model, decay=float(ema_cfg.get("decay", 0.9999)))
    ce_loss = nn.CrossEntropyLoss()
    focal_gamma = float(cfg["training"].get("focal_gamma", 2.0))
    focal_smoothing = float(cfg["training"].get("focal_label_smoothing", 0.1))
    morph_loss_mode = str(cfg["training"].get("morph_loss", "focal"))
    focal_loss = FocalLoss(gamma=focal_gamma, label_smoothing=focal_smoothing)
    base_lr = float(cfg["training"]["lr"])
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    amp_loss_weight = float(cfg["training"].get("amp_loss_weight", 1.0))

    wandb_run = None
    if cfg["wandb"]["enabled"] and _is_main_process():
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg["wandb"]["project"],
                entity=cfg["wandb"]["entity"],
                name=cfg["wandb"]["name"],
                config=cfg,
            )
        except Exception as exc:
            print(f"WandB disabled: {exc}")

    step = 0
    start_epoch = 0
    warm_start_path = cfg["training"].get("warm_start_from")
    resume_path = args.resume or cfg["training"].get("resume_from")
    if resume_path and warm_start_path:
        raise ValueError("Use either training.resume_from for full resume or training.warm_start_from for weights-only init")

    if warm_start_path:
        warm_start_path = Path(warm_start_path)
        loaded, missing, unexpected, shape_mismatch = _load_warm_start_weights(warm_start_path, base_model, device)
        if _is_main_process():
            print(
                f"Warm-start loaded from {warm_start_path}: loaded={loaded} missing={missing} "
                f"unexpected={unexpected} shape_mismatch={shape_mismatch}"
            )

    if resume_path:
        resume_path = Path(resume_path)
        state = _load_checkpoint(resume_path, base_model, optimizer, device, ema)
        step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
        if _is_main_process():
            print(f"Resumed from {resume_path} at epoch={start_epoch} step={step}")
    time_index = int(cfg["validation"]["time_index"])
    default_samples_per_dataset = 1 if val_dataset.num_datasets > 1 else int(cfg["validation"].get("samples", 1))
    samples_per_dataset = int(cfg["validation"].get("samples_per_dataset", default_samples_per_dataset))
    val_indices = val_dataset.get_validation_indices(samples_per_dataset, time_index)
    if not val_indices:
        raise ValueError("No validation indices selected")
    if _is_main_process():
        print(
            f"Validation selection: datasets={val_dataset.dataset_names} "
            f"samples_per_dataset={samples_per_dataset} time_index={time_index}"
        )

    checkpoint_every = int(cfg["training"].get("checkpoint_every", 0))
    max_steps = args.max_steps if args.max_steps is not None else cfg["training"].get("max_steps")

    # calculate total steps for cosine annealing
    total_steps = cfg["training"]["epochs"] * len(train_loader)

    if _is_main_process():
        print("Training.")
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            step += 1
            if max_steps is not None and step > max_steps:
                break
            model.train()
            if warmup_steps > 0 and step <= warmup_steps:
                # Linear Warmup
                current_lr = base_lr * (step / warmup_steps)
            else:
                # Cosine Annealing to zero
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                current_lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
                
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp and amp_dtype is not None
                else nullcontext()
            )
            with autocast_ctx:
                if cfg["dataset"]["token_type"] == "phaedra":

                    amp = batch["amp"]
                    morph = batch["morph"]
                    dataset_ids = batch["dataset_id"].to(device=device, dtype=torch.long)

                    target_amp = amp.view(amp.shape[0], -1)
                    target_morph = morph.view(morph.shape[0], -1)

                    _assert_token_range("amp", amp.view(amp.shape[0], -1), model_cfg.vocab_amp)
                    _assert_token_range("morph", morph.view(morph.shape[0], -1), model_cfg.vocab_morph)

                    amp = amp.to(device)
                    morph = morph.to(device)
                    target_amp = target_amp.to(device)
                    target_morph = target_morph.to(device)

                    amp_pred, logits_morph, mask = model(amp, morph, None, dataset_ids=dataset_ids)

                    # Define mask_flat FIRST so we can use it for indexing
                    mask_flat = mask

                    # --- Gaussian Normalization (Z-score) ---
                    if amp_mean is None or amp_std is None:
                        raise RuntimeError("amp_mean/std not initialized for Phaedra training")

                    # Normalize targets to ~ N(0, 1)
                    target_amp_norm = (target_amp[mask_flat].float() - amp_mean) / amp_std

                    # Calculate losses (amp_pred is naturally near N(0, 1) now)
                    loss_amp = F.l1_loss(amp_pred[mask_flat], target_amp_norm)
                    if morph_loss_mode == "focal":
                        loss_morph = focal_loss(logits_morph[mask_flat], target_morph[mask_flat])
                    else:
                        loss_morph = ce_loss(logits_morph[mask_flat], target_morph[mask_flat])
                    loss = amp_loss_weight * loss_amp + loss_morph

                    # --- De-normalize for metrics ---
                    # Scale predictions back up to the [0, 1024] vocabulary range
                    pred_amp_tokens = (amp_pred * amp_std + amp_mean).round().clamp(0, model_cfg.vocab_amp - 1).long()

                    acc_amp = (pred_amp_tokens[mask_flat] == target_amp[mask_flat]).float().mean().item()
                    acc_morph = (logits_morph.argmax(dim=-1)[mask_flat] == target_morph[mask_flat]).float().mean().item()
                    util_pred_amp = _token_utilization(pred_amp_tokens, model_cfg.vocab_amp, mask_flat)
                    util_pred_morph = _token_utilization(logits_morph.argmax(dim=-1), model_cfg.vocab_morph, mask_flat)
                    util_target_amp = _token_utilization(target_amp, model_cfg.vocab_amp, mask_flat)
                    util_target_morph = _token_utilization(target_morph, model_cfg.vocab_morph, mask_flat)
                else:
                    tokens = batch["tokens"]
                    dataset_ids = batch["dataset_id"].to(device=device, dtype=torch.long)
                    _assert_token_range("fsq", tokens.view(tokens.shape[0], -1), model_cfg.vocab_fsq)
                    tokens = tokens.to(device)
                    logits, mask = model(None, None, tokens, dataset_ids=dataset_ids)
                    target = tokens.view(tokens.shape[0], -1)
                    mask_flat = mask

                    loss = ce_loss(logits[mask_flat], target[mask_flat])
                    acc_amp = None
                    acc_morph = (logits.argmax(dim=-1)[mask_flat] == target[mask_flat]).float().mean().item()
                    util_pred = _token_utilization(logits.argmax(dim=-1), model_cfg.vocab_fsq, mask_flat)
                    util_target = _token_utilization(target, model_cfg.vocab_fsq, mask_flat)

            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                optimizer.step()
            if ema is not None:
                ema.update(base_model)

            if checkpoint_every > 0 and step % checkpoint_every == 0 and _is_main_process():
                ckpt_path = output_dir / f"checkpoint_step_{step}.pt"
                _save_checkpoint(ckpt_path, base_model, optimizer, step, epoch, ema)
                _save_checkpoint(output_dir / "checkpoint_last.pt", base_model, optimizer, step, epoch, ema)

            if step % cfg["training"]["log_every"] == 0 and _is_main_process():
                msg = (
                    f"epoch={epoch} step={step} lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"loss={loss.item():.4f}"
                )
                if acc_amp is not None:
                    msg += (
                        f" loss_amp={loss_amp.item():.4f} loss_morph={loss_morph.item():.4f}"
                        f" acc_amp={acc_amp:.4f} acc_morph={acc_morph:.4f}"
                    )
                    msg += f" util_pred_amp={util_pred_amp:.4f} util_pred_morph={util_pred_morph:.4f}"
                    msg += f" util_target_amp={util_target_amp:.4f} util_target_morph={util_target_morph:.4f}"
                else:
                    msg += f" acc={acc_morph:.4f}"
                    msg += f" util_pred={util_pred:.4f}"
                    msg += f" util_target={util_target:.4f}"
                print(msg)
                if wandb_run:
                    log_payload = {"loss": loss.item()}
                    log_payload["lr"] = optimizer.param_groups[0]["lr"]
                    if acc_amp is not None:
                        log_payload["loss_amp"] = loss_amp.item()
                        log_payload["loss_morph"] = loss_morph.item()
                        log_payload["acc_amp"] = acc_amp
                        log_payload["acc_morph"] = acc_morph
                        log_payload["util_pred_amp"] = util_pred_amp
                        log_payload["util_pred_morph"] = util_pred_morph
                        log_payload["util_target_amp"] = util_target_amp
                        log_payload["util_target_morph"] = util_target_morph
                    else:
                        log_payload["acc"] = acc_morph
                        log_payload["util_pred"] = util_pred
                        log_payload["util_target"] = util_target
                    wandb_run.log(log_payload, step=step)

            if step % cfg["training"]["val_every"] == 0 and _is_main_process():
                val_metrics = {}
                val_metrics_by_dataset: dict[str, dict[str, list[float]]] = {}
                for idx in val_indices:
                    sample = val_dataset[idx]
                    dataset_id = int(sample["dataset_id"])
                    dataset_name = val_dataset.dataset_names[dataset_id]
                    dataset_output_dir = output_dir / "val_plots" / dataset_name
                    val_batch = _batch_sample(sample, device)
                    metrics = validate(
                        base_model,
                        val_batch,
                        cfg,
                        dataset_output_dir,
                        decoder,
                        dataset_label=dataset_name,
                        ema=ema,
                        amp_mean=amp_mean,
                        amp_std=amp_std,
                    )
                    for k, v in metrics.items():
                        val_metrics.setdefault(k, []).append(v)
                        val_metrics_by_dataset.setdefault(dataset_name, {}).setdefault(k, []).append(v)
                val_metrics = {k: float(np.mean(v)) for k, v in val_metrics.items()}
                for dataset_name, dataset_metric_map in val_metrics_by_dataset.items():
                    dataset_key = str(dataset_name).replace(" ", "_").replace("/", "_")
                    for metric_name, values in dataset_metric_map.items():
                        val_metrics[f"val/{dataset_key}/{metric_name}"] = float(np.mean(values))
                if val_metrics:
                    # print(f"Val summary: {val_metrics}")
                    if wandb_run:
                        wandb_run.log(val_metrics, step=step)

        if max_steps is not None and step > max_steps:
            break

    if wandb_run is not None:
        wandb_run.finish()

    if _is_main_process():
        _save_checkpoint(output_dir / "checkpoint_last.pt", base_model, optimizer, step, epoch, ema)

    if _dist_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
