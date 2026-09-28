from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf

from fno_operator.src.dataset import (
    denormalize_fields,
    normalize_fields,
    parse_normalization_stats,
    read_fields,
    resolve_source_dataset_path,
    select_member_indices,
)
from fno_operator.src.metrics import append_metrics, default_metric_store, summarize_metric_store
from fno_operator.src.model import ConditionedFNO2d, ConditionedFNO2dConfig
from hub.eval.plots import save_field_plot
from hub.utils.runtime import configure_torch, get_device


def _load_yaml(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    return cfg


def _resolve_path(config_dir: Path, raw_path: str | None) -> str | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


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
        return payload
    raise ValueError("Unsupported checkpoint payload format")


def _build_schedule_from_fixed_step(input_time_idx: int, final_time_idx: int, step_size: int) -> list[int]:
    if step_size <= 0:
        raise ValueError(f"rollout step must be > 0, got {step_size}")

    schedule = [int(input_time_idx)]
    cur = int(input_time_idx)
    while cur + int(step_size) < int(final_time_idx):
        cur += int(step_size)
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


def _run_rollout(
    model: ConditionedFNO2d,
    start_state_norm: np.ndarray,
    schedule: list[int],
    device: torch.device,
) -> np.ndarray:
    state = torch.from_numpy(start_state_norm).unsqueeze(0).float().to(device)

    for t_in, t_out in zip(schedule[:-1], schedule[1:]):
        lead = torch.tensor([int(t_out - t_in)], dtype=torch.long, device=device)
        with torch.no_grad():
            state = model(state, lead)

    return state[0].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout test for conditioned FNO on continuous CEU-KH")
    parser.add_argument("--config", "--configs", dest="config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config_dir = config_path.parent
    cfg = _load_yaml(config_path)

    configure_torch(tf32=bool(cfg.get("runtime", {}).get("tf32", True)))
    device = get_device()

    ds_cfg = cfg["dataset"]
    input_variables = list(ds_cfg["input_variables"])
    output_variables = list(ds_cfg["output_variables"])
    if input_variables != output_variables:
        raise ValueError("FNO rollout testing requires dataset.input_variables == dataset.output_variables")

    model_cfg = cfg["model"]
    model = ConditionedFNO2d(
        ConditionedFNO2dConfig(
            num_input_vars=len(input_variables),
            num_output_vars=len(output_variables),
            width=int(model_cfg.get("width", 96)),
            depth=int(model_cfg.get("depth", 8)),
            modes_x=int(model_cfg.get("modes_x", 16)),
            modes_y=int(model_cfg.get("modes_y", 16)),
            padding=int(model_cfg.get("padding", 8)),
            max_time_index=int(model_cfg.get("max_time_index", ds_cfg.get("time_end", 14))),
            use_coord_features=bool(model_cfg.get("use_coord_features", True)),
        )
    ).to(device)
    model.eval()

    test_cfg = cfg.get("testing", {}) or {}
    checkpoint_raw = args.checkpoint or test_cfg.get("checkpoint") or test_cfg.get("checkpoint_path")
    if checkpoint_raw is None:
        raise ValueError(
            "Checkpoint path is required. Set testing.checkpoint in the config "
            "or pass --checkpoint on the command line."
        )

    checkpoint_path_str = _resolve_path(config_dir, str(checkpoint_raw))
    assert checkpoint_path_str is not None
    checkpoint_path = Path(checkpoint_path_str).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # released `model.safetensors`, a training checkpoint, or a run directory
    from hub.utils.weights import load_weights
    state_dict, _ = load_weights(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "Checkpoint does not match model architecture: "
            f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
        )

    token_dataset_path = Path(str(_resolve_path(config_dir, str(ds_cfg["token_dataset_path"]))))
    source_dataset_path_cfg = _resolve_path(
        config_dir,
        test_cfg.get("source_fields_path", ds_cfg.get("source_dataset_path")),
    )
    source_dataset_path = resolve_source_dataset_path(
        token_dataset_path=token_dataset_path,
        source_dataset_path=Path(source_dataset_path_cfg) if source_dataset_path_cfg is not None else None,
    )

    norm_stats = parse_normalization_stats(ds_cfg.get("normalization"), output_variables)
    split = str(test_cfg.get("split", ds_cfg.get("split_val", "val")))
    max_members_raw = test_cfg.get("max_members", ds_cfg.get("max_val_members", None))
    max_members = int(max_members_raw) if max_members_raw is not None else None

    input_time_idx = int(test_cfg.get("input_time_idx", ds_cfg.get("fixed_input_time", 0)))
    final_time_idx = int(test_cfg.get("final_time_idx", ds_cfg.get("fixed_output_time", ds_cfg.get("time_end", 14))))
    if final_time_idx <= input_time_idx:
        raise ValueError(
            f"testing.final_time_idx ({final_time_idx}) must be greater than input_time_idx ({input_time_idx})"
        )

    rollouts = _build_rollouts(test_cfg=test_cfg, input_time_idx=input_time_idx, final_time_idx=final_time_idx)
    rollout_names = [spec["name"] for spec in rollouts]

    metric_store: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        rollout_name: {
            "pred_vs_target_recon": default_metric_store(output_variables),
            "pred_vs_true_field": default_metric_store(output_variables),
            "target_recon_vs_true_field": default_metric_store(output_variables),
        }
        for rollout_name in rollout_names
    }

    output_dir_cfg = test_cfg.get("output_dir", str(Path(cfg["validation"]["output_dir"]) / "test"))
    output_dir = Path(str(_resolve_path(config_dir, str(output_dir_cfg)))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    max_plots = int(test_cfg.get("max_plots", cfg.get("validation", {}).get("max_plots", 6)))
    timing_samples = int(test_cfg.get("timing_samples", 1000))
    timed_count = 0
    timed_total_seconds = 0.0
    rollout_plot_counts: dict[str, int] = {name: 0 for name in rollout_names}
    for stale in plots_dir.glob("test_*_sample*_truefield.png"):
        stale.unlink(missing_ok=True)

    with nc.Dataset(str(source_dataset_path), "r") as source_ds:
        member_indices, member_ids = select_member_indices(
            source_ds=source_ds,
            split=split,
            max_members=max_members,
            token_dataset_path=token_dataset_path,
            train_member_start=int(ds_cfg.get("train_member_start", 0)),
            train_member_end=ds_cfg.get("train_member_end", 8000),
            val_samples_hint=ds_cfg.get("val_samples_hint", None),
            test_samples_hint=ds_cfg.get("test_samples_hint", None),
        )

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

            start_raw = read_fields(
                source_ds,
                member_index=member_index,
                time_idx=input_time_idx,
                variables=input_variables,
            )
            target_raw = read_fields(
                source_ds,
                member_index=member_index,
                time_idx=final_time_idx,
                variables=output_variables,
            )

            start_norm = normalize_fields(start_raw, input_variables, norm_stats)
            target_phys = {name: target_raw[i] for i, name in enumerate(output_variables)}
            # Continuous FNO operates directly on fields, so target reconstruction equals target field.
            target_recon = target_phys

            for rollout in rollouts:
                do_timing = timing_samples > 0 and timed_count < timing_samples
                if do_timing and device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter() if do_timing else 0.0

                pred_norm = _run_rollout(
                    model=model,
                    start_state_norm=start_norm,
                    schedule=rollout["schedule"],
                    device=device,
                )
                pred_phys = denormalize_fields(pred_norm, output_variables, norm_stats)

                if do_timing:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    timed_total_seconds += float(time.perf_counter() - t0)
                    timed_count += 1

                if max_plots > 0 and rollout_plot_counts[rollout["name"]] < max_plots:
                    plot_idx = rollout_plot_counts[rollout["name"]]
                    true_field_t = torch.from_numpy(
                        np.stack([target_phys[name] for name in output_variables], axis=0)
                    ).float()
                    pred_phys_t = torch.from_numpy(
                        np.stack([pred_phys[name] for name in output_variables], axis=0)
                    ).float()
                    save_field_plot(
                        path=str(
                            plots_dir
                            / (
                                f"test_{rollout['name']}_sample{plot_idx:03d}_member{int(member_id)}_truefield.png"
                            )
                        ),
                        target_field=true_field_t,
                        pred_field=pred_phys_t,
                    )
                    rollout_plot_counts[rollout["name"]] += 1

                append_metrics(
                    metric_store=metric_store[rollout["name"]]["pred_vs_target_recon"],
                    variables=output_variables,
                    pred_map=pred_phys,
                    target_map=target_recon,
                )
                append_metrics(
                    metric_store=metric_store[rollout["name"]]["pred_vs_true_field"],
                    variables=output_variables,
                    pred_map=pred_phys,
                    target_map=target_phys,
                )
                append_metrics(
                    metric_store=metric_store[rollout["name"]]["target_recon_vs_true_field"],
                    variables=output_variables,
                    pred_map=target_recon,
                    target_map=target_phys,
                )

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "token_dataset": str(token_dataset_path),
        "source_fields_dataset": str(source_dataset_path),
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
            "pred_vs_target_recon": summarize_metric_store(metric_store[name]["pred_vs_target_recon"]),
            "pred_vs_true_field": summarize_metric_store(metric_store[name]["pred_vs_true_field"]),
            "target_recon_vs_true_field": summarize_metric_store(metric_store[name]["target_recon_vs_true_field"]),
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
