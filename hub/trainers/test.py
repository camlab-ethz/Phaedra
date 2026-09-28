from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf

from hub.config import load_config
from hub.eval.plots import save_paper_triplet_plot, save_sample_overview_plot, save_field_plot
from hub.models import build_model
from hub.models.diffusion_process import MaskedDiscreteDiffusion
from hub.utils.phaedra_decoder import decode_tokens, load_token_decoder
from hub.utils.runtime import configure_torch, get_device


def _load_yaml(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if all(k.startswith("model.") for k in keys):
        return {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_model_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported checkpoint format: expected dict-like model state")


def _relative_l1(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.mean(np.abs(pred - target)))
    den = float(np.mean(np.abs(target)) + 1e-8)
    return num / den


def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.sqrt(np.mean((pred - target) ** 2)))
    den = float(np.sqrt(np.mean(target ** 2)) + 1e-8)
    return num / den


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


def _build_denorm_lookup_from_config(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, tuple[float, float]]], dict[str, str]]:
    denorm_cfg = cfg.get("denormalization", {}) or {}

    aliases_raw = denorm_cfg.get("name_aliases", {}) or {}
    name_aliases = {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases_raw.items()}

    lookup: dict[str, dict[str, tuple[float, float]]] = {}
    datasets = denorm_cfg.get("datasets", []) or []
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        raw_name = entry.get("name")
        if raw_name is None:
            continue
        key = str(raw_name).strip().lower()

        vars_out = entry.get("field_variables_out") or entry.get("output_variables") or entry.get("variables") or []
        means = entry.get("_normalization_mean") or entry.get("normalization_mean") or entry.get("mean") or []
        stds = entry.get("_normalization_std") or entry.get("normalization_std") or entry.get("std") or []

        if len(vars_out) != len(means) or len(vars_out) != len(stds):
            raise ValueError(
                f"Denormalization entry '{raw_name}' has inconsistent lengths: "
                f"vars={len(vars_out)} means={len(means)} stds={len(stds)}"
            )

        lookup[key] = {
            str(var_name): (float(mean_val), float(std_val) if float(std_val) != 0.0 else 1.0)
            for var_name, mean_val, std_val in zip(vars_out, means, stds)
        }

    return lookup, name_aliases


def _source_dataset_stem_from_tokens(token_dataset_path: Path) -> str | None:
    try:
        with nc.Dataset(str(token_dataset_path), "r") as ds:
            if "source_dataset" not in ds.ncattrs():
                return None
            raw = str(ds.getncattr("source_dataset"))
        return Path(raw).stem.strip().lower()
    except Exception:
        return None


def _source_dataset_path_from_tokens(token_dataset_path: Path) -> Path | None:
    try:
        with nc.Dataset(str(token_dataset_path), "r") as ds:
            if "source_dataset" not in ds.ncattrs():
                return None
            raw = str(ds.getncattr("source_dataset"))
        path = Path(raw).expanduser()
        return path if path.exists() else None
    except Exception:
        return None


def _resolve_denorm_stats_for_dataset(
    dataset_name: str,
    token_dataset_path: Path,
    output_variables: list[str],
    denorm_lookup: dict[str, dict[str, tuple[float, float]]],
    name_aliases: dict[str, str],
) -> tuple[dict[str, tuple[float, float]], str]:
    candidates: list[str] = [
        str(dataset_name).strip().lower(),
        token_dataset_path.stem.strip().lower(),
    ]

    source_stem = _source_dataset_stem_from_tokens(token_dataset_path)
    if source_stem:
        candidates.append(source_stem)

    for candidate in candidates:
        key = name_aliases.get(candidate, candidate)
        var_stats = denorm_lookup.get(key)
        if var_stats is None:
            continue
        resolved = {var_name: var_stats.get(var_name, (0.0, 1.0)) for var_name in output_variables}
        return resolved, key

    fallback = {var_name: (0.0, 1.0) for var_name in output_variables}
    return fallback, "identity"


def _load_member_tokens_phaedra(
    ds: nc.Dataset,
    member_index: int,
    time_idx: int,
    variables: list[str],
    morph_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    morph = [np.asarray(ds.variables[f"{var}_morph"][member_index, time_idx], dtype=np.int64) for var in variables]
    amp = [np.asarray(ds.variables[f"{var}_amp"][member_index, time_idx], dtype=np.int64) for var in variables]

    morph_t = torch.from_numpy(np.stack(morph, axis=0)).long()
    amp_t = torch.from_numpy(np.stack(amp, axis=0)).long()
    if morph_offset:
        morph_t = morph_t - int(morph_offset)
    return amp_t, morph_t


def _load_member_tokens_fsq(
    ds: nc.Dataset,
    member_index: int,
    time_idx: int,
    variables: list[str],
) -> torch.Tensor:
    tokens = [np.asarray(ds.variables[f"{var}_tokens"][member_index, time_idx], dtype=np.int64) for var in variables]
    return torch.from_numpy(np.stack(tokens, axis=0)).long()


def _select_member_indices(ds: nc.Dataset, split: str, max_members: int | None) -> tuple[list[int], list[int]]:
    if "member" in ds.variables:
        member_values = np.asarray(ds.variables["member"][:], dtype=np.int64)
    else:
        member_values = np.arange(int(ds.dimensions["member"].size), dtype=np.int64)

    split_attr = f"split_{split}_range"
    if split_attr in ds.ncattrs():
        start_str, end_str = str(ds.getncattr(split_attr)).split(":")
        start = int(start_str)
        end = int(end_str)
        member_indices = [i for i, member_id in enumerate(member_values) if start <= int(member_id) < end]
    else:
        member_indices = list(range(len(member_values)))

    if max_members is not None:
        member_indices = member_indices[: int(max_members)]

    if not member_indices:
        raise ValueError(f"No members found for split={split}")

    member_ids = [int(member_values[i]) for i in member_indices]
    return member_indices, member_ids


def _build_schedule_from_fixed_step(input_time_idx: int, final_time_idx: int, step_size: int) -> list[int]:
    if step_size <= 0:
        raise ValueError(f"rollout step must be > 0, got {step_size}")
    schedule = [int(input_time_idx)]
    cur = int(input_time_idx)
    while cur + step_size < int(final_time_idx):
        cur += step_size
        schedule.append(cur)
    if schedule[-1] != int(final_time_idx):
        schedule.append(int(final_time_idx))
    return schedule


def _build_schedule_from_step_list(input_time_idx: int, final_time_idx: int, steps: list[int]) -> list[int]:
    if not steps:
        raise ValueError("rollout steps list cannot be empty")

    schedule = [int(input_time_idx)]
    cur = int(input_time_idx)
    for step in steps:
        if int(step) <= 0:
            raise ValueError(f"rollout step values must be > 0, got {step}")
        cur += int(step)
        schedule.append(cur)

    if schedule[-1] != int(final_time_idx):
        raise ValueError(
            f"steps-based rollout must end exactly at final_time_idx={final_time_idx}, got schedule={schedule}"
        )
    return schedule


def _validate_schedule(name: str, schedule: list[int], input_time_idx: int, final_time_idx: int) -> None:
    if len(schedule) < 2:
        raise ValueError(f"rollout '{name}' must contain at least two time points")
    if schedule[0] != int(input_time_idx) or schedule[-1] != int(final_time_idx):
        raise ValueError(
            f"rollout '{name}' must start at {input_time_idx} and end at {final_time_idx}, got {schedule}"
        )
    if any(schedule[i + 1] <= schedule[i] for i in range(len(schedule) - 1)):
        raise ValueError(f"rollout '{name}' schedule must be strictly increasing, got {schedule}")


def _build_rollouts(test_cfg: dict[str, Any], input_time_idx: int, final_time_idx: int) -> list[dict[str, Any]]:
    rollouts_cfg = test_cfg.get("rollouts")
    if rollouts_cfg is None:
        rollouts_cfg = [
            {
                "name": "direct_0_to_14",
                "display_name": "Direct 0->14",
                "schedule": [int(input_time_idx), int(final_time_idx)],
            },
            {
                "name": "step_2_rollout",
                "display_name": "Step-2 Rollout",
                "step": 2,
            },
            {
                "name": "step_6_rollout",
                "display_name": "Step-6 Rollout",
                "step": 6,
            },
        ]

    out: list[dict[str, Any]] = []
    for item in rollouts_cfg:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid rollout spec (expected dict): {item}")

        name = str(item.get("name", f"rollout_{len(out)}"))
        display_name = str(item.get("display_name", name))

        if "schedule" in item:
            schedule = [int(v) for v in item["schedule"]]
        elif "steps" in item:
            raw_steps = item["steps"]
            if isinstance(raw_steps, int):
                schedule = _build_schedule_from_fixed_step(input_time_idx, final_time_idx, int(raw_steps))
            else:
                schedule = _build_schedule_from_step_list(
                    input_time_idx,
                    final_time_idx,
                    [int(v) for v in list(raw_steps)],
                )
        elif "step" in item:
            schedule = _build_schedule_from_fixed_step(input_time_idx, final_time_idx, int(item["step"]))
        else:
            raise ValueError(f"rollout '{name}' requires one of: schedule, step, steps")

        _validate_schedule(name, schedule, input_time_idx, final_time_idx)
        out.append({"name": name, "display_name": display_name, "schedule": schedule})

    return out


def _decode_normalized_fields(
    decoder,
    token_type: str,
    *,
    amp_tokens: torch.Tensor | None = None,
    morph_tokens: torch.Tensor | None = None,
    fsq_tokens: torch.Tensor | None = None,
) -> np.ndarray:
    with torch.no_grad():
        recon = decode_tokens(
            decoder,
            fsq_tokens=fsq_tokens.to(decoder.device) if fsq_tokens is not None else None,
            amp_tokens=amp_tokens.to(decoder.device) if amp_tokens is not None else None,
            morph_tokens=morph_tokens.to(decoder.device) if morph_tokens is not None else None,
        )

    arr = recon.detach().float().cpu().numpy()
    if arr.ndim == 4:
        return arr[:, 0].astype(np.float32)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError(f"Unexpected decoded tensor shape: {arr.shape}")


def _clip_tokens_for_decoder_phaedra(
    amp_tokens: torch.Tensor,
    morph_tokens: torch.Tensor,
    amp_vocab_size: int,
    morph_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    amp_clip = amp_tokens.long().clamp(min=0, max=int(amp_vocab_size) - 1)
    morph_clip = morph_tokens.long().clamp(min=0, max=int(morph_vocab_size) - 1)
    return amp_clip, morph_clip


def _clip_tokens_for_decoder_fsq(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return tokens.long().clamp(min=0, max=int(vocab_size) - 1)


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


def _build_single_step_sample(
    current_amp: torch.Tensor,
    current_morph: torch.Tensor,
    output_variables: list[str],
    member_id: int,
    t_in: int,
    t_out: int,
) -> dict[str, Any]:
    out_shape = (len(output_variables), int(current_amp.shape[-2]), int(current_amp.shape[-1]))
    return {
        "input_morph": current_morph.detach().cpu().long(),
        "input_amp": current_amp.detach().cpu().long(),
        "output_morph": torch.zeros(out_shape, dtype=torch.long),
        "output_amp": torch.zeros(out_shape, dtype=torch.long),
        "member_idx": int(member_id),
        "input_time_idx": int(t_in),
        "output_time_idx": int(t_out),
        "lead_time_idx": int(t_out - t_in),
        "dataset_name": "ceu_kh",
        "var_names": tuple(output_variables),
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=False)
        else:
            out[k] = v
    return out


def _predict_step_seq2seq_phaedra(
    model,
    current_amp: torch.Tensor,
    current_morph: torch.Tensor,
    t_in: int,
    t_out: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        pred_amp, pred_morph = model.predict(
            input_amp=current_amp.unsqueeze(0).to(device),
            input_morph=current_morph.unsqueeze(0).to(device),
            input_time_idx=torch.tensor([int(t_in)], dtype=torch.long, device=device),
            output_time_idx=torch.tensor([int(t_out)], dtype=torch.long, device=device),
            input_var_ids=None,
            output_var_ids=None,
            input_var_mask=None,
            output_var_mask=None,
            problem_type_id=torch.zeros((1,), dtype=torch.long, device=device),
        )

    vars_out = int(current_amp.shape[0])
    h = int(current_amp.shape[1])
    w = int(current_amp.shape[2])
    return pred_amp.view(1, vars_out, h, w)[0], pred_morph.view(1, vars_out, h, w)[0]


def _predict_step_seq2seq_fsq(
    model,
    current_tokens: torch.Tensor,
    t_in: int,
    t_out: int,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        pred_tokens = model.predict(
            input_tokens=current_tokens.unsqueeze(0).to(device),
            input_time_idx=torch.tensor([int(t_in)], dtype=torch.long, device=device),
            output_time_idx=torch.tensor([int(t_out)], dtype=torch.long, device=device),
            input_var_ids=None,
            output_var_ids=None,
            input_var_mask=None,
            output_var_mask=None,
            problem_type_id=torch.zeros((1,), dtype=torch.long, device=device),
        )

    vars_out = int(current_tokens.shape[0])
    h = int(current_tokens.shape[1])
    w = int(current_tokens.shape[2])
    return pred_tokens.view(1, vars_out, h, w)[0]


def _predict_step_diffusion(
    model,
    diffusion: MaskedDiscreteDiffusion,
    sample: dict[str, Any],
    collator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = _to_device(collator([sample]), device)
    with torch.no_grad():
        pred_morph_seq, pred_amp_seq = diffusion.generate(
            model=model,
            seq_morph=batch["sequence_morph"],
            seq_amp=batch["sequence_amp"],
            attention_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            variable_ids=batch["variable_ids"],
            spatial_indices=batch["spatial_indices"],
            lead_time=batch["lead_time"],
            pde_type_id=batch["problem_type_id"],
            target_nonpad_mask=batch["target_nonpad_mask"],
        )

    vars_out = int(batch["target_amp_grid"].shape[1])
    h = int(batch["grid_height"].item())
    w = int(batch["grid_width"].item())
    start = int(batch["target_start"].item())
    count = int(vars_out * h * w)

    pred_amp = pred_amp_seq[0, start : start + count].view(vars_out, h, w)
    pred_morph = pred_morph_seq[0, start : start + count].view(vars_out, h, w)
    return pred_amp, pred_morph


def _predict_step_hybrid(
    model,
    current_amp: torch.Tensor,
    current_morph: torch.Tensor,
    output_variables: list[str],
    t_in: int,
    t_out: int,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    variable_order = list(cfg["dataset"]["variable_order"])
    var_to_slot = {name: idx for idx, name in enumerate(variable_order)}

    missing_vars = [name for name in output_variables if name not in var_to_slot]
    if missing_vars:
        raise ValueError(f"Variables missing from dataset.variable_order: {missing_vars}")

    max_variables = int(cfg["model"].get("max_variables", len(variable_order)))
    h = int(current_amp.shape[1])
    w = int(current_amp.shape[2])

    source_amp = torch.zeros((1, max_variables, h, w), dtype=torch.long, device=device)
    source_morph = torch.zeros((1, max_variables, h, w), dtype=torch.long, device=device)
    active_var_mask = torch.zeros((1, max_variables), dtype=torch.bool, device=device)
    slot_var_ids = torch.zeros((1, max_variables), dtype=torch.long, device=device)

    for local_idx, name in enumerate(output_variables):
        slot = int(var_to_slot[name])
        source_amp[0, slot] = current_amp[local_idx].to(device)
        source_morph[0, slot] = current_morph[local_idx].to(device)
        active_var_mask[0, slot] = True
        slot_var_ids[0, slot] = slot + 1

    with torch.no_grad():
        pred_amp_flat, pred_morph_flat = model.predict_tokens(
            source_amp=source_amp,
            source_morph=source_morph,
            lead_time_idx=torch.tensor([int(t_out - t_in)], dtype=torch.long, device=device),
            active_var_mask=active_var_mask,
            slot_var_ids=slot_var_ids,
            spatial_mask=torch.ones((1, h, w), dtype=torch.bool, device=device),
        )

    pred_amp_slots = pred_amp_flat.view(1, h, w, max_variables).permute(0, 3, 1, 2).contiguous()[0]
    pred_morph_slots = pred_morph_flat.view(1, h, w, max_variables).permute(0, 3, 1, 2).contiguous()[0]

    pred_amp = torch.stack([pred_amp_slots[var_to_slot[name]] for name in output_variables], dim=0)
    pred_morph = torch.stack([pred_morph_slots[var_to_slot[name]] for name in output_variables], dim=0)
    return pred_amp, pred_morph


def _run_rollout_phaedra(
    cfg: dict[str, Any],
    model,
    diffusion: MaskedDiscreteDiffusion | None,
    diffusion_collator,
    start_amp: torch.Tensor,
    start_morph: torch.Tensor,
    output_variables: list[str],
    member_id: int,
    schedule: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_amp = start_amp.clone().to(device)
    current_morph = start_morph.clone().to(device)

    for t_in, t_out in zip(schedule[:-1], schedule[1:]):
        if cfg["model_type"] == "seq2seq":
            current_amp, current_morph = _predict_step_seq2seq_phaedra(
                model=model,
                current_amp=current_amp,
                current_morph=current_morph,
                t_in=t_in,
                t_out=t_out,
                device=device,
            )
        elif cfg["model_type"] == "diffusion":
            if diffusion is None:
                raise RuntimeError("diffusion rollout requested but diffusion helper is not initialized")

            sample = _build_single_step_sample(
                current_amp=current_amp,
                current_morph=current_morph,
                output_variables=output_variables,
                member_id=member_id,
                t_in=t_in,
                t_out=t_out,
            )
            current_amp, current_morph = _predict_step_diffusion(
                model=model,
                diffusion=diffusion,
                sample=sample,
                collator=diffusion_collator,
                device=device,
            )
        else:
            current_amp, current_morph = _predict_step_hybrid(
                model=model,
                current_amp=current_amp,
                current_morph=current_morph,
                output_variables=output_variables,
                t_in=t_in,
                t_out=t_out,
                cfg=cfg,
                device=device,
            )

    return current_amp, current_morph


def _run_rollout_fsq(
    model,
    start_tokens: torch.Tensor,
    schedule: list[int],
    device: torch.device,
) -> torch.Tensor:
    current_tokens = start_tokens.clone().to(device)
    for t_in, t_out in zip(schedule[:-1], schedule[1:]):
        current_tokens = _predict_step_seq2seq_fsq(
            model=model,
            current_tokens=current_tokens,
            t_in=t_in,
            t_out=t_out,
            device=device,
        )
    return current_tokens


def _default_metric_store(variables: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        "relative_l1": {name: [] for name in variables},
        "relative_l2": {name: [] for name in variables},
        "w1": {name: [] for name in variables},
    }


def _append_metrics(
    metric_store: dict[str, dict[str, list[float]]],
    variables: list[str],
    pred_map: dict[str, np.ndarray],
    target_map: dict[str, np.ndarray],
) -> None:
    for name in variables:
        pred_arr = pred_map[name]
        target_arr = target_map[name]
        metric_store["relative_l1"][name].append(_relative_l1(pred_arr, target_arr))
        metric_store["relative_l2"][name].append(_relative_l2(pred_arr, target_arr))
        metric_store["w1"][name].append(_wasserstein_1d(pred_arr, target_arr))


def _summarize_metric_store(metric_store: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name, per_var in metric_store.items():
        per_var_mean = {
            var_name: (float(np.mean(values)) if values else float("nan"))
            for var_name, values in per_var.items()
        }
        per_var_median = {
            var_name: (float(np.median(values)) if values else float("nan"))
            for var_name, values in per_var.items()
        }

        mean_values = [v for v in per_var_mean.values() if np.isfinite(v)]
        median_values = [v for v in per_var_median.values() if np.isfinite(v)]
        summary[metric_name] = {
            "mean_per_variable": per_var_mean,
            "median_per_variable": per_var_median,
            "mean_across_variables": float(np.mean(mean_values)) if mean_values else float("nan"),
            "median_across_variables": float(np.mean(median_values)) if median_values else float("nan"),
        }
    return summary


def _build_member_id_to_source_index(source_ds: nc.Dataset) -> dict[int, int]:
    if "member" not in source_ds.variables:
        return {}
    vals = np.asarray(source_ds.variables["member"][:], dtype=np.int64)
    return {int(member_id): int(i) for i, member_id in enumerate(vals)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified rollout test entrypoint")
    parser.add_argument("--config", "--configs", dest="config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    token_type = str(cfg.get("dataset", {}).get("token_type", "phaedra")).lower().strip()
    configure_torch(tf32=bool(cfg.get("runtime", {}).get("tf32", True)))
    device = get_device()

    if token_type not in {"phaedra", "fsq"}:
        raise RuntimeError(f"Unsupported token_type for testing: {token_type}")
    if token_type == "fsq" and str(cfg.get("model_type")) != "seq2seq":
        raise RuntimeError("FSQ token testing currently supports model_type=seq2seq only")

    test_cfg = cfg.get("testing", {}) or {}
    checkpoint_raw = args.checkpoint or test_cfg.get("checkpoint") or test_cfg.get("checkpoint_path")
    if checkpoint_raw is None:
        raise ValueError(
            "Checkpoint path is required. Set testing.checkpoint in the config "
            "or pass --checkpoint on the command line."
        )
    checkpoint_path = Path(str(checkpoint_raw)).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(cfg).to(device)
    model.eval()

    # released `model.safetensors`, a training checkpoint, or a run directory
    from hub.utils.weights import load_weights
    state_dict, _ = load_weights(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "Checkpoint does not match model architecture: "
            f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
        )
    print(
        "[test] "
        f"checkpoint={checkpoint_path} "
        f"loaded={len(model.state_dict()) - len(load_result.missing_keys)} "
        f"missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}"
    )

    decoder = None
    if bool(cfg["validation"].get("decoder_enabled", False)):
        decoder = load_token_decoder(
            {
                "enabled": True,
                "phaedra_root": cfg["validation"].get("phaedra_root"),
                "model_name": cfg["validation"]["model_name"],
                "model_path": cfg["validation"]["model_path"],
                "model_config": cfg["validation"].get("model_config"),
                "use_ema": bool(cfg["validation"].get("use_ema", True)),
            },
            device=device,
        )
    if decoder is None:
        raise RuntimeError("validation.decoder_enabled=true is required for rollout reconstruction testing")

    diffusion = None
    diffusion_collator = None
    if cfg["model_type"] == "diffusion":
        if token_type != "phaedra":
            raise RuntimeError("Diffusion testing currently supports token_type=phaedra only")
        amp_mask_id = int(cfg["dataset"]["amp_vocab_size"]) + 2
        morph_mask_id = int(cfg["dataset"]["morph_vocab_size"]) + 2
        diffusion = MaskedDiscreteDiffusion(
            num_steps=int(cfg["model"].get("diffusion_steps", cfg.get("diffusion", {}).get("num_steps", 32))),
            morph_mask_id=morph_mask_id,
            amp_mask_id=amp_mask_id,
        )
        from hub.data.collators import build_collator

        diffusion_collator = build_collator("diffusion", cfg["dataset"], cfg["model"])

    input_variables = list(cfg["dataset"]["input_variables"])
    output_variables = list(cfg["dataset"]["output_variables"])
    if input_variables != output_variables:
        raise ValueError("Rollout testing currently requires dataset.input_variables == dataset.output_variables")
    amp_vocab_size = int(cfg["dataset"].get("amp_vocab_size", 0))
    morph_vocab_size = int(cfg["dataset"].get("morph_vocab_size", 0))
    fsq_vocab_size = int(cfg["dataset"].get("fsq_vocab_size", 0))

    input_time_idx = int(test_cfg.get("input_time_idx", cfg["dataset"].get("fixed_input_time", 0)))
    final_time_idx = int(test_cfg.get("final_time_idx", cfg["dataset"].get("fixed_output_time", cfg["dataset"].get("time_end", 14))))
    if final_time_idx <= input_time_idx:
        raise ValueError(
            f"testing.final_time_idx ({final_time_idx}) must be greater than input_time_idx ({input_time_idx})"
        )

    split = str(test_cfg.get("split", cfg["dataset"].get("split_val", "val")))
    max_members_raw = test_cfg.get("max_members", cfg["dataset"].get("max_val_members", None))
    max_members = int(max_members_raw) if max_members_raw is not None else None

    rollouts = _build_rollouts(test_cfg=test_cfg, input_time_idx=input_time_idx, final_time_idx=final_time_idx)
    rollout_names = [spec["name"] for spec in rollouts]

    token_dataset_path = Path(cfg["dataset"]["path"]).expanduser().resolve()
    if not token_dataset_path.exists():
        raise FileNotFoundError(f"Token dataset not found: {token_dataset_path}")

    source_fields_path_cfg = test_cfg.get("source_fields_path")
    source_fields_path = None
    if source_fields_path_cfg:
        source_fields_path = Path(str(source_fields_path_cfg)).expanduser().resolve()
    else:
        source_fields_path = _source_dataset_path_from_tokens(token_dataset_path)
    if source_fields_path is None or not source_fields_path.exists():
        raise FileNotFoundError(
            "Could not resolve source fields dataset path. Set testing.source_fields_path in config "
            "or provide source_dataset attribute in the token netCDF."
        )

    # Physical-space (mean, std) per dataset: shipped in evaluation/denormalization.yaml
    # (the same statistics the tokens were generated with); override per config.
    from hub.utils.runtime import resolve_repo_path
    denorm_cfg_path = test_cfg.get("denormalization_config") or "evaluation/denormalization.yaml"
    denorm_source_cfg = _load_yaml(resolve_repo_path(denorm_cfg_path))
    denorm_lookup, denorm_aliases = _build_denorm_lookup_from_config(denorm_source_cfg)

    denorm_stats, denorm_key = _resolve_denorm_stats_for_dataset(
        dataset_name=str(test_cfg.get("dataset_name", "ceu_kh")),
        token_dataset_path=token_dataset_path,
        output_variables=output_variables,
        denorm_lookup=denorm_lookup,
        name_aliases=denorm_aliases,
    )
    print(f"[test] denormalization_key={denorm_key} stats={denorm_stats}")

    metric_store: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        rollout_name: {
            "pred_vs_target_recon": _default_metric_store(output_variables),
            "pred_vs_true_field": _default_metric_store(output_variables),
            "target_recon_vs_true_field": _default_metric_store(output_variables),
        }
        for rollout_name in rollout_names
    }

    output_dir = Path(test_cfg.get("output_dir", Path(cfg["validation"]["output_dir"]) / "test")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    max_plots = int(test_cfg.get("max_plots", cfg.get("validation", {}).get("max_plots", 6)))
    timing_samples = int(test_cfg.get("timing_samples", 1000))
    timed_count = 0
    timed_total_seconds = 0.0
    rollout_plot_counts: dict[str, int] = {name: 0 for name in rollout_names}
    for stale in plots_dir.glob("test_*_sample*.png"):
        stale.unlink(missing_ok=True)

    with nc.Dataset(str(token_dataset_path), "r") as token_ds, nc.Dataset(str(source_fields_path), "r") as source_ds:
        if token_type == "phaedra":
            morph_offset = int(token_ds.getncattr("morphology_offset")) if "morphology_offset" in token_ds.ncattrs() else 0
        else:
            morph_offset = 0
            if fsq_vocab_size <= 0:
                if "codebook_size" not in token_ds.ncattrs():
                    raise ValueError("FSQ vocab size not set in config or dataset attributes")
                fsq_vocab_size = int(token_ds.getncattr("codebook_size"))
        member_indices, member_ids = _select_member_indices(token_ds, split=split, max_members=max_members)
        member_id_to_source_index = _build_member_id_to_source_index(source_ds)
        rollout_desc = ", ".join([f"{x['name']}:{x['schedule']}" for x in rollouts])

        print(
            "[test] "
            f"split={split} members={len(member_indices)} "
            f"input_time={input_time_idx} final_time={final_time_idx} "
            f"rollouts={{{rollout_desc}}}"
        )

        for pos, (member_index, member_id) in enumerate(zip(member_indices, member_ids)):
            if pos % 20 == 0:
                print(f"[test] member {pos + 1}/{len(member_indices)} (id={member_id})")

            if token_type == "phaedra":
                start_amp, start_morph = _load_member_tokens_phaedra(
                    token_ds,
                    member_index=member_index,
                    time_idx=input_time_idx,
                    variables=input_variables,
                    morph_offset=morph_offset,
                )
                target_amp, target_morph = _load_member_tokens_phaedra(
                    token_ds,
                    member_index=member_index,
                    time_idx=final_time_idx,
                    variables=output_variables,
                    morph_offset=morph_offset,
                )
                target_amp, target_morph = _clip_tokens_for_decoder_phaedra(
                    amp_tokens=target_amp,
                    morph_tokens=target_morph,
                    amp_vocab_size=amp_vocab_size,
                    morph_vocab_size=morph_vocab_size,
                )

                target_norm = _decode_normalized_fields(
                    decoder=decoder,
                    token_type=token_type,
                    amp_tokens=target_amp.to(device),
                    morph_tokens=target_morph.to(device),
                )
            else:
                start_tokens = _load_member_tokens_fsq(
                    token_ds,
                    member_index=member_index,
                    time_idx=input_time_idx,
                    variables=input_variables,
                )
                target_tokens = _load_member_tokens_fsq(
                    token_ds,
                    member_index=member_index,
                    time_idx=final_time_idx,
                    variables=output_variables,
                )
                target_tokens = _clip_tokens_for_decoder_fsq(target_tokens, fsq_vocab_size)
                target_norm = _decode_normalized_fields(
                    decoder=decoder,
                    token_type=token_type,
                    fsq_tokens=target_tokens.to(device),
                )

            target_phys = _denormalize_fields(
                normalized_fields=target_norm,
                var_names=output_variables,
                stats=denorm_stats,
            )

            source_member_index = member_id_to_source_index.get(int(member_id), int(member_id))
            true_field: dict[str, np.ndarray] = {
                name: np.asarray(source_ds.variables[name][source_member_index, final_time_idx, :, :], dtype=np.float32)
                for name in output_variables
            }

            for rollout in rollouts:
                do_timing = timing_samples > 0 and timed_count < timing_samples
                if do_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter() if do_timing else 0.0

                if token_type == "phaedra":
                    pred_amp, pred_morph = _run_rollout_phaedra(
                        cfg=cfg,
                        model=model,
                        diffusion=diffusion,
                        diffusion_collator=diffusion_collator,
                        start_amp=start_amp,
                        start_morph=start_morph,
                        output_variables=output_variables,
                        member_id=int(member_id),
                        schedule=rollout["schedule"],
                        device=device,
                    )
                    pred_amp, pred_morph = _clip_tokens_for_decoder_phaedra(
                        amp_tokens=pred_amp,
                        morph_tokens=pred_morph,
                        amp_vocab_size=amp_vocab_size,
                        morph_vocab_size=morph_vocab_size,
                    )

                    pred_norm = _decode_normalized_fields(
                        decoder=decoder,
                        token_type=token_type,
                        amp_tokens=pred_amp,
                        morph_tokens=pred_morph,
                    )
                else:
                    pred_tokens = _run_rollout_fsq(
                        model=model,
                        start_tokens=start_tokens,
                        schedule=rollout["schedule"],
                        device=device,
                    )
                    pred_tokens = _clip_tokens_for_decoder_fsq(pred_tokens, fsq_vocab_size)
                    pred_norm = _decode_normalized_fields(
                        decoder=decoder,
                        token_type=token_type,
                        fsq_tokens=pred_tokens,
                    )
                pred_phys = _denormalize_fields(
                    normalized_fields=pred_norm,
                    var_names=output_variables,
                    stats=denorm_stats,
                )

                if do_timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timed_total_seconds += float(time.perf_counter() - t0)
                    timed_count += 1

                if max_plots > 0 and rollout_plot_counts[rollout["name"]] < max_plots:
                    plot_idx = rollout_plot_counts[rollout["name"]]
                    pred_norm_t = torch.from_numpy(pred_norm).float()
                    target_norm_t = torch.from_numpy(target_norm).float()

                    if token_type == "phaedra":
                        plot_target_morph = target_morph
                        plot_target_amp = target_amp
                        plot_pred_morph = pred_morph
                        plot_pred_amp = pred_amp
                    else:
                        plot_target_morph = target_tokens
                        plot_target_amp = target_tokens
                        plot_pred_morph = pred_tokens
                        plot_pred_amp = pred_tokens

                    save_sample_overview_plot(
                        path=str(
                            plots_dir
                            / (
                                f"test_{rollout['name']}_sample{plot_idx:03d}_member{int(member_id)}.png"
                            )
                        ),
                        target_morph=plot_target_morph,
                        pred_morph=plot_pred_morph,
                        target_amp=plot_target_amp,
                        pred_amp=plot_pred_amp,
                        target_field=target_norm_t,
                        pred_field=pred_norm_t,
                        var_names=output_variables,
                    )

                    true_field_t = torch.from_numpy(
                        np.stack([true_field[name] for name in output_variables], axis=0)
                    ).float()
                    pred_phys_t = torch.from_numpy(
                        np.stack([pred_phys[name] for name in output_variables], axis=0)
                    ).float()
                    save_paper_triplet_plot(
                        path=str(
                            plots_dir
                            / (
                                f"test_{rollout['name']}_sample{plot_idx:03d}_member{int(member_id)}_truefield.png"
                            )
                        ),
                        ground_truth_field=true_field_t,
                        recon_from_gt_tokens=torch.from_numpy(
                            np.stack([target_phys[name] for name in output_variables], axis=0)
                        ).float(),
                        recon_from_pred_tokens=pred_phys_t,
                    )
                    save_field_plot(
                        str(
                            plots_dir
                            / (
                                f"test_{rollout['name']}_sample{plot_idx:03d}_member{int(member_id)}_truefield.png"
                            )
                        ),
                        true_field_t,
                        pred_phys_t
                    )


                    rollout_plot_counts[rollout["name"]] += 1

                _append_metrics(
                    metric_store[rollout["name"]]["pred_vs_target_recon"],
                    variables=output_variables,
                    pred_map=pred_phys,
                    target_map=target_phys,
                )
                _append_metrics(
                    metric_store[rollout["name"]]["pred_vs_true_field"],
                    variables=output_variables,
                    pred_map=pred_phys,
                    target_map=true_field,
                )
                _append_metrics(
                    metric_store[rollout["name"]]["target_recon_vs_true_field"],
                    variables=output_variables,
                    pred_map=target_phys,
                    target_map=true_field,
                )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "token_dataset": str(token_dataset_path),
        "source_fields_dataset": str(source_fields_path),
        "split": split,
        "max_members": max_members,
        "input_time_idx": input_time_idx,
        "final_time_idx": final_time_idx,
        "rollouts": {spec["name"]: spec["schedule"] for spec in rollouts},
        "metrics": {},
    }

    avg_seconds_per_sample = float(timed_total_seconds / timed_count) if timed_count > 0 else float("nan")
    samples_per_second = float(timed_count / timed_total_seconds) if timed_total_seconds > 0.0 else float("nan")
    summary["timing"] = {
        "requested_samples": int(timing_samples),
        "measured_samples": int(timed_count),
        "total_seconds": float(timed_total_seconds),
        "avg_seconds_per_sample": avg_seconds_per_sample,
        "samples_per_second": samples_per_second,
    }

    for rollout in rollouts:
        name = rollout["name"]
        summary["metrics"][name] = {
            "display_name": rollout["display_name"],
            "schedule": rollout["schedule"],
            "pred_vs_target_recon": _summarize_metric_store(metric_store[name]["pred_vs_target_recon"]),
            "pred_vs_true_field": _summarize_metric_store(metric_store[name]["pred_vs_true_field"]),
            "target_recon_vs_true_field": _summarize_metric_store(metric_store[name]["target_recon_vs_true_field"]),
        }

    summary_path = output_dir / "test_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        f"checkpoint={summary['checkpoint']}",
        f"token_dataset={summary['token_dataset']}",
        f"source_fields_dataset={summary['source_fields_dataset']}",
        f"split={split}",
        f"input_time_idx={input_time_idx}",
        f"final_time_idx={final_time_idx}",
        (
            "timing="
            f"requested={summary['timing']['requested_samples']} "
            f"measured={summary['timing']['measured_samples']} "
            f"total_seconds={summary['timing']['total_seconds']:.6f} "
            f"avg_seconds_per_sample={summary['timing']['avg_seconds_per_sample']:.6f} "
            f"samples_per_second={summary['timing']['samples_per_second']:.6f}"
        ),
    ]
    for rollout in rollouts:
        name = rollout["name"]
        lines.append(f"\n[{name}] {rollout['display_name']} schedule={rollout['schedule']}")
        for ref_key in ["pred_vs_target_recon", "pred_vs_true_field", "target_recon_vs_true_field"]:
            lines.append(f"  - {ref_key}")
            for metric_name in ["relative_l1", "relative_l2", "w1"]:
                aggregate = summary["metrics"][name][ref_key][metric_name]["mean_across_variables"]
                lines.append(f"      {metric_name}_mean_across_variables={aggregate:.8f}")

    log_path = output_dir / "test_metrics_log.txt"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"[test] wrote summary to {summary_path}")
    print(f"[test] wrote log to {log_path}")
    if max_plots > 0:
        print(f"[test] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
