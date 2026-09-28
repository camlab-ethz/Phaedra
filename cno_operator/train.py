from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cno_operator.src.model import (
    ContinuousCNO2d,
    ContinuousCNO2dConfig,
    assert_parameter_budget,
    count_parameters,
)
from fno_operator.src.dataset import ContinuousKHAll2AllConfig, ContinuousKHAll2AllDataset
from hub.utils.checkpoint import load_checkpoint, save_checkpoint
from hub.utils.runtime import configure_torch, get_device, set_seed
from hub.utils.wandb_utils import grad_norm_l2, max_abs_grad, maybe_init_wandb


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


def _dist_info() -> tuple[bool, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return False, 0, 1, local_rank, True

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, rank, world_size, local_rank, rank == 0


def _reduce_mean(value: float, device: torch.device, distributed: bool, world_size: int) -> float:
    if not distributed or world_size <= 1:
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / float(world_size)).item())


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Resume path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Resume path must be a file or directory: {path}")

    last_ckpt = path / "checkpoint_last.pt"
    if last_ckpt.exists():
        return last_ckpt

    step_ckpts = sorted(
        path.glob("checkpoint_step_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if step_ckpts:
        return step_ckpts[0]

    best_ckpt = path / "checkpoint_best.pt"
    if best_ckpt.exists():
        return best_ckpt

    raise FileNotFoundError(f"No checkpoint file found in resume directory: {path}")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _save_validation_plot(
    path: Path,
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    output_variables: list[str],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_np = pred_phys.detach().float().cpu().numpy()
    target_np = target_phys.detach().float().cpu().numpy()
    diff_np = pred_np - target_np

    n_vars = int(pred_np.shape[0])
    fig, axes = plt.subplots(n_vars, 3, figsize=(9.5, max(2.6 * n_vars, 3.5)), squeeze=False)

    for row in range(n_vars):
        vname = output_variables[row] if row < len(output_variables) else f"var_{row}"
        vmin = min(float(target_np[row].min()), float(pred_np[row].min()))
        vmax = max(float(target_np[row].max()), float(pred_np[row].max()))
        dmax = max(float(abs(diff_np[row]).max()), 1.0e-8)

        im_t = axes[row, 0].imshow(target_np[row], cmap="turbo", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"GT {vname}")
        axes[row, 0].axis("off")
        fig.colorbar(im_t, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im_p = axes[row, 1].imshow(pred_np[row], cmap="turbo", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Pred {vname}")
        axes[row, 1].axis("off")
        fig.colorbar(im_p, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im_d = axes[row, 2].imshow(diff_np[row], cmap="bwr", vmin=-dmax, vmax=dmax)
        axes[row, 2].set_title(f"Pred-GT {vname}")
        axes[row, 2].axis("off")
        fig.colorbar(im_d, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


@torch.no_grad()
def _run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_variables: list[str],
    norm_stats: dict[str, tuple[float, float]],
    max_batches: int,
    max_plots: int,
    plots_dir: Path,
    step: int,
) -> dict[str, float]:
    model.eval()

    loss_sum = 0.0
    rel_l1_sum = 0.0
    per_var_sum = {name: 0.0 for name in output_variables}
    count = 0
    plots_written = 0

    do_plots = int(max_plots) > 0
    can_plot = False
    if do_plots:
        try:
            import matplotlib  # noqa: F401

            can_plot = True
            plots_dir.mkdir(parents=True, exist_ok=True)
            for stale in plots_dir.glob("val_step*_sample*.png"):
                stale.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[warn] validation plotting disabled (matplotlib unavailable): {exc}")
            can_plot = False

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= int(max_batches):
            break

        batch = _to_device(batch, device)
        pred = model(batch["input_fields"], batch["lead_time_idx"])
        target = batch["target_fields"]
        loss = F.l1_loss(pred, target)

        loss_sum += float(loss.item())

        batch_var_rels: list[float] = []
        for var_idx, var_name in enumerate(output_variables):
            mean_val, std_val = norm_stats[var_name]
            pred_phys = pred[:, var_idx] * float(std_val) + float(mean_val)
            target_phys = target[:, var_idx] * float(std_val) + float(mean_val)
            rel = torch.mean(torch.abs(pred_phys - target_phys)) / (torch.mean(torch.abs(target_phys)) + 1.0e-8)
            rel_v = float(rel.item())
            per_var_sum[var_name] += rel_v
            batch_var_rels.append(rel_v)

        if batch_var_rels:
            rel_l1_sum += float(sum(batch_var_rels) / len(batch_var_rels))

        if can_plot and plots_written < int(max_plots):
            bsz = int(pred.shape[0])
            for sample_idx in range(bsz):
                if plots_written >= int(max_plots):
                    break

                pred_phys_vars = []
                target_phys_vars = []
                for var_idx, var_name in enumerate(output_variables):
                    mean_val, std_val = norm_stats[var_name]
                    pred_phys_vars.append(pred[sample_idx, var_idx] * float(std_val) + float(mean_val))
                    target_phys_vars.append(target[sample_idx, var_idx] * float(std_val) + float(mean_val))

                pred_phys = torch.stack(pred_phys_vars, dim=0)
                target_phys = torch.stack(target_phys_vars, dim=0)

                _save_validation_plot(
                    path=plots_dir / f"val_step{step}_sample{plots_written}.png",
                    pred_phys=pred_phys,
                    target_phys=target_phys,
                    output_variables=output_variables,
                    title=(
                        f"step={step} batch={batch_idx} sample={sample_idx} "
                        f"lead_time={int(batch['lead_time_idx'][sample_idx].item())}"
                    ),
                )
                plots_written += 1

        count += 1

    model.train()

    if count == 0:
        return {"val/loss_l1_norm": float("nan"), "val/rel_l1_phys": float("nan")}

    metrics: dict[str, float] = {
        "val/loss_l1_norm": float(loss_sum / count),
        "val/rel_l1_phys": float(rel_l1_sum / count),
    }
    for name in output_variables:
        metrics[f"val/rel_l1_phys_{name}"] = float(per_var_sum[name] / count)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditioned CNO on continuous CEU-KH fields")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(config_path)
    config_dir = config_path.parent

    ds_cfg = cfg["dataset"]
    # Original convention required all_even_forward (0, 14, 2) for RKH-style
    # time-evolving experiments. Loosened here so the same trainer also
    # handles `fixed` (single-pair time-evolving runs, e.g. the small-data
    # benchmark) and `identity` (steady-state Poisson / Darcy).
    valid_pair_modes = {"all_even_forward", "fixed", "identity"}
    pair_mode_train = str(ds_cfg.get("pair_mode_train", ""))
    if pair_mode_train not in valid_pair_modes:
        raise ValueError(
            f"pair_mode_train={pair_mode_train!r} not supported. "
            f"Choose from {sorted(valid_pair_modes)}."
        )

    set_seed(int(cfg["runtime"].get("seed", 42)))
    configure_torch(tf32=bool(cfg["runtime"].get("tf32", True)))

    distributed, rank, world_size, local_rank, is_main = _dist_info()
    device = get_device()

    if is_main:
        device_name = str(device)
        if device.type == "cuda":
            device_name = f"cuda:{torch.cuda.current_device()}"
        print(
            "[startup] "
            f"rank={rank}/{world_size} "
            f"local_rank={local_rank} "
            f"pid={os.getpid()} "
            f"host={os.uname().nodename} "
            f"device={device_name}",
            flush=True,
        )

    token_dataset_path = _resolve_path(config_dir, str(ds_cfg["token_dataset_path"]))
    source_dataset_path = _resolve_path(config_dir, ds_cfg.get("source_dataset_path"))

    train_ds = ContinuousKHAll2AllDataset(
        ContinuousKHAll2AllConfig(
            token_dataset_path=str(token_dataset_path),
            source_dataset_path=source_dataset_path,
            split=str(ds_cfg.get("split_train", "train")),
            input_variables=list(ds_cfg["input_variables"]),
            output_variables=list(ds_cfg["output_variables"]),
            pair_mode=str(ds_cfg.get("pair_mode_train", "all_even_forward")),
            pair_time_start=int(ds_cfg.get("time_start", 0)),
            pair_time_end=int(ds_cfg.get("time_end", 14)),
            pair_time_step=int(ds_cfg.get("time_step", 2)),
            fixed_input_time=int(ds_cfg.get("fixed_input_time", 0)),
            fixed_output_time=int(ds_cfg.get("fixed_output_time", 14)),
            max_members=ds_cfg.get("max_train_members"),
            train_member_start=int(ds_cfg.get("train_member_start", 0)),
            train_member_end=ds_cfg.get("train_member_end", 8000),
            val_samples_hint=ds_cfg.get("val_samples_hint", None),
            test_samples_hint=ds_cfg.get("test_samples_hint", None),
            normalization=ds_cfg.get("normalization"),
        )
    )

    val_ds = ContinuousKHAll2AllDataset(
        ContinuousKHAll2AllConfig(
            token_dataset_path=str(token_dataset_path),
            source_dataset_path=source_dataset_path,
            split=str(ds_cfg.get("split_val", "val")),
            input_variables=list(ds_cfg["input_variables"]),
            output_variables=list(ds_cfg["output_variables"]),
            pair_mode=str(ds_cfg.get("pair_mode_val", "fixed")),
            pair_time_start=int(ds_cfg.get("time_start", 0)),
            pair_time_end=int(ds_cfg.get("time_end", 14)),
            pair_time_step=int(ds_cfg.get("time_step", 2)),
            fixed_input_time=int(ds_cfg.get("fixed_input_time", 0)),
            fixed_output_time=int(ds_cfg.get("fixed_output_time", 14)),
            max_members=ds_cfg.get("max_val_members"),
            train_member_start=int(ds_cfg.get("train_member_start", 0)),
            train_member_end=ds_cfg.get("train_member_end", 8000),
            val_samples_hint=ds_cfg.get("val_samples_hint", None),
            test_samples_hint=ds_cfg.get("test_samples_hint", None),
            normalization=ds_cfg.get("normalization"),
        )
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            drop_last=False,
            seed=int(cfg["runtime"].get("seed", 42)),
        )

    num_workers = int(ds_cfg.get("num_workers", 4))
    pin_memory = bool(ds_cfg.get("pin_memory", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(ds_cfg["batch_size_train"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(ds_cfg["batch_size_val"]),
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 1),
    )

    model_cfg = cfg["model"]
    base_model = ContinuousCNO2d(
        ContinuousCNO2dConfig(
            num_input_vars=len(ds_cfg["input_variables"]),
            num_output_vars=len(ds_cfg["output_variables"]),
            base_width=int(model_cfg.get("base_width", 82)),
            num_levels=int(model_cfg.get("num_levels", 4)),
            blocks_per_level=int(model_cfg.get("blocks_per_level", 2)),
            bottleneck_blocks=int(model_cfg.get("bottleneck_blocks", 2)),
            channel_multiplier=int(model_cfg.get("channel_multiplier", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            max_time_index=int(model_cfg.get("max_time_index", ds_cfg.get("time_end", 14))),
            use_coord_features=bool(model_cfg.get("use_coord_features", True)),
        )
    ).to(device)

    assert_parameter_budget(
        base_model,
        target_params=int(cfg["budget"]["target_params"]),
        tolerance=int(cfg["budget"]["tolerance"]),
    )

    model: torch.nn.Module = base_model
    if distributed:
        if device.type == "cuda":
            model = DDP(base_model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        else:
            model = DDP(base_model)

    total_params = count_parameters(base_model)
    if is_main:
        print(
            "[train] "
            f"world_size={world_size} "
            f"params_total={total_params} ({total_params / 1e6:.2f}M) "
            f"train_samples={len(train_ds)} "
            f"val_samples={len(val_ds)} "
            f"time_pairs_train={len(train_ds.time_pairs)} "
            f"time_pairs_val={len(val_ds.time_pairs)}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 1.0e-4)),
        betas=(0.9, 0.95),
    )

    mixed_precision = str(cfg["training"].get("mixed_precision", "bf16")).lower()
    use_amp = device.type == "cuda" and mixed_precision in {"bf16", "fp16"}
    amp_dtype = None
    scaler = None
    if use_amp and mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    elif use_amp and mixed_precision == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()

    wb = maybe_init_wandb(cfg) if is_main else None

    output_dir = Path(str(_resolve_path(config_dir, str(cfg["validation"]["output_dir"]))))
    plots_dir = output_dir / "plots"
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    step = 0
    start_epoch = 0
    resume_raw = args.resume or cfg["training"].get("resume_from")
    if resume_raw:
        resume_candidate = Path(str(_resolve_path(config_dir, str(resume_raw))))
        resume_path = _resolve_resume_checkpoint(resume_candidate)
        state = load_checkpoint(str(resume_path), device=device)

        _base_model(model).load_state_dict(state["model"], strict=True)
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])

        start_epoch = int(state.get("epoch", 0)) + 1
        step = int(state.get("step", 0))
        if is_main:
            print(f"[resume] checkpoint={resume_path} start_epoch={start_epoch} step={step}")

    epochs = int(cfg["training"]["epochs"])
    if start_epoch >= epochs:
        if is_main:
            print(f"[train] checkpoint already reached epochs={epochs}; nothing to run")
        train_ds.close()
        val_ds.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return

    steps_per_epoch = len(train_loader)
    # Optional step-based cap (small-data benchmark). Same semantics as the
    # seq2seq trainer: when set, LR schedule uses max_steps as the cosine
    # denominator and training exits cleanly at step==max_steps.
    max_steps = int(cfg["training"].get("max_steps", 0))
    if max_steps > 0:
        total_steps = int(max_steps)
    else:
        total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(cfg["training"].get("warmup_steps", 0))
    log_every = int(cfg["training"].get("log_every", 20))
    val_every = int(cfg["training"].get("val_every", 200))
    checkpoint_every = int(cfg["training"].get("checkpoint_every", 1000))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    base_lr = float(cfg["training"]["lr"])

    output_variables = list(ds_cfg["output_variables"])
    norm_stats = {
        name: (
            float(ds_cfg["normalization"][name]["mean"]),
            float(ds_cfg["normalization"][name]["std"]),
        )
        for name in output_variables
    }

    early_stop = False
    try:
        for epoch in range(start_epoch, epochs):
            if max_steps > 0 and step >= max_steps:
                if is_main:
                    print(f"[train] max_steps={max_steps} reached at step={step}; "
                          f"stopping before epoch {epoch}", flush=True)
                early_stop = True
                break

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                if max_steps > 0 and step >= max_steps:
                    early_stop = True
                    break
                step += 1
                batch = _to_device(batch, device)

                lr = _cosine_lr(step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=base_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if use_amp and amp_dtype is not None
                    else nullcontext()
                )
                with autocast_ctx:
                    pred = model(batch["input_fields"], batch["lead_time_idx"])
                    loss = F.l1_loss(pred, batch["target_fields"])

                if scaler is not None:
                    scaler.scale(loss).backward()
                    grad_pre = grad_norm_l2(model.parameters())
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    grad_post = grad_norm_l2(model.parameters())
                    grad_abs = max_abs_grad(model.parameters())
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_pre = grad_norm_l2(model.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    grad_post = grad_norm_l2(model.parameters())
                    grad_abs = max_abs_grad(model.parameters())
                    optimizer.step()

                if step % log_every == 0:
                    train_log = {
                        "train/loss_l1_norm": _reduce_mean(float(loss.item()), device, distributed, world_size),
                        "train/lr": float(lr),
                        "train/grad_norm_pre_clip": _reduce_mean(float(grad_pre), device, distributed, world_size),
                        "train/grad_norm_post_clip": _reduce_mean(float(grad_post), device, distributed, world_size),
                        "train/grad_max_abs": _reduce_mean(float(grad_abs), device, distributed, world_size),
                    }
                    if is_main:
                        print(
                            f"epoch={epoch} step={step} loss_l1_norm={train_log['train/loss_l1_norm']:.6f} "
                            f"lr={train_log['train/lr']:.2e} "
                            f"grad_pre={train_log['train/grad_norm_pre_clip']:.4f} "
                            f"grad_post={train_log['train/grad_norm_post_clip']:.4f}"
                        )
                        if wb is not None:
                            wb.log(train_log, step=step)

                if val_every > 0 and step % val_every == 0:
                    if is_main:
                        val_metrics = _run_validation(
                            model=_base_model(model),
                            val_loader=val_loader,
                            device=device,
                            output_variables=output_variables,
                            norm_stats=norm_stats,
                            max_batches=int(cfg["validation"].get("max_batches", 16)),
                            max_plots=int(cfg["validation"].get("max_plots", 6)),
                            plots_dir=plots_dir,
                            step=step,
                        )
                        print(f"[val] step={step} {val_metrics}")
                        if wb is not None:
                            wb.log(val_metrics, step=step)
                    if distributed:
                        dist.barrier()

                if checkpoint_every > 0 and step % checkpoint_every == 0:
                    if is_main:
                        save_checkpoint(
                            path=str(output_dir / f"checkpoint_step_{step}.pt"),
                            model=_base_model(model),
                            optimizer=optimizer,
                            scheduler=None,
                            epoch=epoch,
                            step=step,
                            cfg=cfg,
                        )
                        save_checkpoint(
                            path=str(output_dir / "checkpoint_last.pt"),
                            model=_base_model(model),
                            optimizer=optimizer,
                            scheduler=None,
                            epoch=epoch,
                            step=step,
                            cfg=cfg,
                        )
                    if distributed:
                        dist.barrier()

        if is_main:
            save_checkpoint(
                path=str(output_dir / "checkpoint_last.pt"),
                model=_base_model(model),
                optimizer=optimizer,
                scheduler=None,
                epoch=epochs - 1,
                step=step,
                cfg=cfg,
            )

            run_summary = {
                "epochs": epochs,
                "steps": step,
                "world_size": world_size,
                "dataset_token_path": str(token_dataset_path),
                "dataset_source_path": str(train_ds.source_dataset_path),
                "time_pairs_train": train_ds.time_pairs,
                "time_pairs_val": val_ds.time_pairs,
                "params_total": total_params,
            }
            with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(run_summary, handle, indent=2)

    finally:
        train_ds.close()
        val_ds.close()

        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
