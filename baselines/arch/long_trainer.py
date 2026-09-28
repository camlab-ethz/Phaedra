"""Long-training trainer for the 6 seq2seq benchmark variants.

Multi-epoch, DDP-capable training intended for a convergence comparison run on
the full RKH dataset. Inherits the same speedups as `bench_trainer.py` (bf16
autocast, FlashSDPA, fused AdamW, `torch.compile`, persistent dataloader
workers) plus DDP gradient bucketing.

Validation runs each epoch with token-grid plots written to
`<output_dir>/plots/<variant>_latest_sample{i}.png`. The "latest" filename is
overwritten every epoch so you always see the most recent prediction without
the directory blowing up.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from baselines.arch.config_io import load_config
from baselines.arch.data import build_seq2seq_dataloaders
from baselines.arch.models import BUILDERS
from hub.eval.plots import save_token_plot
from hub.utils.checkpoint import load_checkpoint, save_checkpoint
from hub.utils.runtime import configure_torch, get_device, set_seed
from hub.utils.wandb_utils import maybe_init_wandb


# ---------------------------------------------------------------------------
# Distributed plumbing.
# ---------------------------------------------------------------------------
def _dist_info() -> tuple[bool, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return False, 0, 1, local_rank, True
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), local_rank, dist.get_rank() == 0


def _reduce_mean(value: float, device: torch.device, distributed: bool, world_size: int) -> float:
    if not distributed or world_size <= 1:
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / float(world_size)).item())


def _base(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        model = model.module
    inner = getattr(model, "_orig_mod", None)
    return inner if inner is not None else model


# ---------------------------------------------------------------------------
# Training step + helpers.
# ---------------------------------------------------------------------------
def _cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if warmup > 0 and step <= warmup:
        return base_lr * (step / max(1, warmup))
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _focal_ce(logits: torch.Tensor, targets: torch.Tensor,
              gamma: float, alpha: float | None, label_smoothing: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    if alpha is not None:
        loss = loss * float(alpha)
    return loss.mean()


def _forward(model: torch.nn.Module, batch: dict[str, Any], mode: str):
    out = model(
        input_amp=batch["input_amp"],
        input_morph=batch["input_morph"],
        input_time_idx=batch["input_time_idx"],
        output_time_idx=batch["output_time_idx"],
        input_var_ids=batch["input_var_ids"],
        output_var_ids=batch["output_var_ids"],
        input_var_mask=batch["input_var_mask"],
        output_var_mask=batch["output_var_mask"],
        problem_type_id=batch["problem_type_id"],
        target_amp=batch["output_amp"],
        mode=mode,
    )
    bsz, V, H, W = batch["output_amp"].shape
    tgt_amp_flat = batch["output_amp"].view(bsz, V * H * W)
    tgt_morph_flat = batch["output_morph"].view(bsz, V * H * W)
    return out, tgt_amp_flat, tgt_morph_flat


def _step(model, batch, mode, focal_gamma, focal_alpha, label_smoothing):
    out, tgt_amp_flat, tgt_morph_flat = _forward(model, batch, mode)
    loss_amp = _focal_ce(
        out.amp_logits.reshape(-1, out.amp_logits.shape[-1]),
        tgt_amp_flat.reshape(-1),
        gamma=focal_gamma, alpha=focal_alpha, label_smoothing=label_smoothing,
    )
    loss_morph = _focal_ce(
        out.morph_logits.reshape(-1, out.morph_logits.shape[-1]),
        tgt_morph_flat.reshape(-1),
        gamma=focal_gamma, alpha=focal_alpha, label_smoothing=label_smoothing,
    )
    loss = loss_amp + loss_morph
    return {
        "out": out, "loss": loss,
        "loss_amp": loss_amp.detach(), "loss_morph": loss_morph.detach(),
    }


def _pred_to_grid(pred_flat: torch.Tensor, target_grid: torch.Tensor) -> torch.Tensor:
    bsz, V, H, W = target_grid.shape
    return pred_flat.view(bsz, V, H, W).contiguous()


def _token_accuracy(pred_flat: torch.Tensor, target_grid: torch.Tensor,
                    var_mask: torch.Tensor) -> float:
    bsz, V, H, W = target_grid.shape
    pred_grid = _pred_to_grid(pred_flat, target_grid)
    active = var_mask[:, :, None, None].expand(bsz, V, H, W)
    if not active.any():
        return 0.0
    return float((pred_grid[active] == target_grid[active]).float().mean().item())


def _save_val_plots_overwrite(
    out, batch, plot_dir: Path, plots_written: int, max_plots: int, variant: str,
) -> int:
    """Save with the `<variant>_latest_sample{i}.png` filename pattern. Files
    from previous epochs are overwritten implicitly via the fixed names; no
    accumulated `_step123` suffix.
    """
    if plots_written >= max_plots:
        return plots_written
    pred_amp_grid = _pred_to_grid(out.pred_amp, batch["output_amp"])
    pred_morph_grid = _pred_to_grid(out.pred_morph, batch["output_morph"])
    bsz = int(batch["output_amp"].shape[0])
    for sample_idx in range(bsz):
        if plots_written >= max_plots:
            break
        active = batch["output_var_mask"][sample_idx]
        if not bool(active.any()):
            continue
        save_token_plot(
            path=str(plot_dir / f"{variant}_latest_sample{plots_written}.png"),
            target_morph=batch["output_morph"][sample_idx][active].detach().cpu(),
            pred_morph=pred_morph_grid[sample_idx][active].detach().cpu(),
            target_amp=batch["output_amp"][sample_idx][active].detach().cpu(),
            pred_amp=pred_amp_grid[sample_idx][active].detach().cpu(),
        )
        plots_written += 1
    return plots_written


@torch.no_grad()
def _run_eval(
    model, val_loader, device, autocast_ctx, max_batches: int,
    focal_gamma: float, focal_alpha: float | None, label_smoothing: float,
    plot_dir: Path | None, max_plots: int, variant: str, save_plots: bool,
) -> tuple[dict[str, float], float, int]:
    model.eval()
    sums = {"val/loss": 0.0, "val/loss_amp": 0.0, "val/loss_morph": 0.0,
            "val/token_acc_amp": 0.0, "val/token_acc_morph": 0.0}
    n = 0
    samples = 0
    plots_written = 0
    if save_plots and plot_dir is not None and max_plots > 0:
        plot_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        batch = _to_device(batch, device)
        with autocast_ctx():
            step_result = _step(model, batch, "B", focal_gamma, focal_alpha, label_smoothing)
        out = step_result["out"]
        sums["val/loss"] += float(step_result["loss"].item())
        sums["val/loss_amp"] += float(step_result["loss_amp"].item())
        sums["val/loss_morph"] += float(step_result["loss_morph"].item())
        sums["val/token_acc_amp"] += _token_accuracy(out.pred_amp, batch["output_amp"], batch["output_var_mask"])
        sums["val/token_acc_morph"] += _token_accuracy(out.pred_morph, batch["output_morph"], batch["output_var_mask"])
        samples += int(batch["input_amp"].shape[0])
        n += 1
        if save_plots and plot_dir is not None and max_plots > 0:
            plots_written = _save_val_plots_overwrite(out, batch, plot_dir, plots_written, max_plots, variant)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    metrics = {k: float(v / max(n, 1)) for k, v in sums.items()}
    model.train()
    return metrics, (t1 - t0), samples


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Long-training trainer for the seq2seq variants (DDP)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--variant", type=str, required=True, choices=sorted(BUILDERS.keys()))
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a previous run (loads model + optimizer + step + epoch).")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Load model weights only from a checkpoint (transfer-learning init). "
                             "Optimizer/step/epoch start fresh. Shape-mismatched keys are skipped "
                             "with a printed warning rather than failing. Ignored if --resume is set.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_variant"] = args.variant

    set_seed(int(cfg["runtime"]["seed"]))
    configure_torch(tf32=bool(cfg["runtime"].get("tf32", True)))

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            # Disable the math backend so SDPA can't silently fall back to the
            # fp32-attention-matrix path under torch.compile (which materialises
            # a [B*H, L, L] tensor and OOMs at seq=4096, batch=32).
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass

    device = get_device()
    distributed, rank, world_size, local_rank, is_main = _dist_info()

    if is_main:
        print(f"[startup] variant={args.variant} rank={rank}/{world_size} "
              f"local_rank={local_rank} pid={os.getpid()} host={os.uname().nodename} device={device}", flush=True)

    train_loader, val_loader, train_sampler = build_seq2seq_dataloaders(
        cfg, distributed=distributed, rank=rank, world_size=world_size,
    )

    builder = BUILDERS[args.variant]
    model = builder(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[budget] variant={args.variant} params={total_params} ({total_params / 1e6:.2f}M)", flush=True)

    # ---- pretrained init (transfer-learning) -----------------------------
    # Distinct from --resume: only model weights are loaded, optimizer and
    # LR schedule start fresh. Keys with mismatched shapes are skipped (not
    # an error) -- this lets you swap datasets that share the encoder/decoder
    # but differ in, e.g., variable count or vocab size.
    if args.pretrained and not args.resume:
        state = load_checkpoint(str(args.pretrained), device=device)
        ckpt_sd = state["model"]
        model_sd = model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        shape_mm: list[tuple[str, tuple, tuple]] = []
        for k, v in ckpt_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered[k] = v
            elif k in model_sd:
                shape_mm.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        not_in_ckpt = [k for k in model_sd if k not in ckpt_sd]
        not_in_model = [k for k in ckpt_sd if k not in model_sd]
        incompatible = model.load_state_dict(filtered, strict=False)
        if is_main:
            print(f"[pretrained] src={args.pretrained}", flush=True)
            print(f"[pretrained]   loaded={len(filtered)}/{len(ckpt_sd)} keys "
                  f"(model_total={len(model_sd)})", flush=True)
            if shape_mm:
                print(f"[pretrained]   shape_mismatch={len(shape_mm)} (kept random init), e.g.:", flush=True)
                for k, cs, ms in shape_mm[:3]:
                    print(f"[pretrained]     - {k}: ckpt={cs} vs model={ms}", flush=True)
            if not_in_ckpt:
                print(f"[pretrained]   missing_from_ckpt={len(not_in_ckpt)} (kept random init), e.g.:", flush=True)
                for k in not_in_ckpt[:3]:
                    print(f"[pretrained]     - {k}", flush=True)
            if not_in_model:
                print(f"[pretrained]   ignored_from_ckpt={len(not_in_model)} (not in current model), e.g.:", flush=True)
                for k in not_in_model[:3]:
                    print(f"[pretrained]     - {k}", flush=True)
            if incompatible.unexpected_keys:
                # Should be empty since we pre-filtered, but log defensively.
                print(f"[pretrained]   unexpected={len(incompatible.unexpected_keys)}", flush=True)
    elif args.pretrained and args.resume and is_main:
        print(f"[pretrained] WARN: --pretrained={args.pretrained} ignored because --resume is set", flush=True)

    # Eagerly build any flex_attention BlockMasks on this device, BEFORE the
    # outer torch.compile + DDP wrap. Lazy in-forward creation would graph-break
    # dynamo and the flex_attention call would fall back to the eager unfused
    # path (with the warning "flex_attention called without torch.compile()").
    _n_setup = 0
    for sub in model.modules():
        if hasattr(sub, "setup_block_mask") and callable(sub.setup_block_mask):
            sub.setup_block_mask(device)
            _n_setup += 1
    if is_main and _n_setup > 0:
        print(f"[flex] pre-built BlockMask on {_n_setup} submodules", flush=True)

    use_bf16 = str(cfg["training"].get("mixed_precision", "bf16")).lower() == "bf16"

    def autocast_ctx():
        if device.type == "cuda" and use_bf16:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.amp.autocast(device_type="cuda", enabled=False)

    do_compile = bool(cfg["training"].get("compile", True))
    if do_compile:
        compile_mode = str(cfg["training"].get("compile_mode", "default"))
        if is_main:
            print(f"[compile] mode={compile_mode}", flush=True)
        model = torch.compile(model, mode=compile_mode, fullgraph=False)

    if distributed:
        if device.type == "cuda":
            model = DDP(
                model, device_ids=[device.index], output_device=device.index,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        else:
            model = DDP(model)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    epochs = int(cfg["training"]["epochs"])
    steps_per_epoch = len(train_loader)
    # Optional step-based cap for small-data runs where epoch counts don't
    # reflect the desired training intensity. When set:
    #   - LR cosine schedule denominator becomes max_steps (not epochs * spe)
    #   - training breaks out of the batch loop when step >= max_steps
    #   - `epochs` is then a sentinel upper bound (set it high in the config)
    max_steps = int(cfg["training"].get("max_steps", 0))
    if max_steps > 0:
        total_steps = int(max_steps)
    else:
        total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(cfg["training"].get("warmup_steps", 2000))
    base_lr = float(cfg["training"]["lr"])
    grad_clip = float(cfg["training"]["grad_clip"])
    log_every = int(cfg["training"].get("log_every", 50))
    # Gradient accumulation: lets a memory-heavy config keep the reference
    # GLOBAL batch (per_gpu_bs * world_size * accum) on fewer/smaller GPUs.
    grad_accum = max(1, int(cfg["training"].get("grad_accum_steps", 1)))
    focal_gamma = float(cfg["training"].get("focal_gamma", 2.0))
    focal_alpha = cfg["training"].get("focal_alpha", None)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    val_every_epoch = int(cfg["training"].get("val_every_epoch", 1))
    ckpt_every_epoch = int(cfg["training"].get("checkpoint_every_epoch", 10))
    plot_every_epoch = int(cfg["training"].get("plot_every_epoch", 1))
    mode_cycle_raw = cfg["training"].get("mode_cycle", ["A"])
    mode_cycle = [str(m).upper() for m in (mode_cycle_raw if isinstance(mode_cycle_raw, list) else [mode_cycle_raw])]
    if not mode_cycle or any(m not in {"A", "B"} for m in mode_cycle):
        raise ValueError(f"training.mode_cycle must contain only A/B, got {mode_cycle}")

    per_gpu_bs = int(cfg["dataset"]["batch_size_train"])
    if grad_accum > 1:
        total_steps = max(1, total_steps // grad_accum)
    if is_main and grad_accum > 1:
        print(f"[train] grad_accum={grad_accum} -> effective global batch "
              f"{per_gpu_bs * world_size * grad_accum}; total optimizer steps={total_steps}",
              flush=True)
    if is_main:
        print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
              f"steps_per_epoch={steps_per_epoch} per_gpu_bs={per_gpu_bs} global_bs={per_gpu_bs * world_size}", flush=True)
        cap_str = f" max_steps={max_steps} (cap)" if max_steps > 0 else ""
        print(f"[train] epochs={epochs} total_steps={total_steps}{cap_str} "
              f"warmup={warmup_steps} mode_cycle={mode_cycle}", flush=True)

    output_dir = Path(cfg["validation"]["output_dir"]).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    wb = maybe_init_wandb(cfg) if is_main else None

    step = 0
    start_epoch = 0
    resume_batch_idx = 0
    if args.resume:
        state = load_checkpoint(str(args.resume), device=device)
        _base(model).load_state_dict(state["model"], strict=True)
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        step = int(state.get("step", 0))
        saved_epoch = int(state.get("epoch", 0))
        saved_batch_idx = int(state.get("batch_idx", -1))
        if saved_batch_idx + 1 >= steps_per_epoch:
            start_epoch = saved_epoch + 1
        else:
            start_epoch = saved_epoch
            resume_batch_idx = saved_batch_idx + 1
        if is_main:
            print(f"[resume] step={step} start_epoch={start_epoch} resume_batch_idx={resume_batch_idx}", flush=True)

    if start_epoch >= epochs:
        if is_main:
            print(f"[train] start_epoch={start_epoch} already >= epochs={epochs}; exit", flush=True)
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return

    model.train()
    micro_idx = 0          # gradient-accumulation position within an update

    for epoch in range(start_epoch, epochs):
        # max_steps early stop: check at the top of each epoch so the post-loop
        # final-val + final-checkpoint section still runs once when we exit.
        if max_steps > 0 and step >= max_steps:
            if is_main:
                print(f"[train] max_steps={max_steps} reached at step={step}; "
                      f"stopping before epoch {epoch}", flush=True)
            break
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        mode = mode_cycle[epoch % len(mode_cycle)]
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < resume_batch_idx:
                continue
            # max_steps early stop: check before the step so we exit cleanly
            # after exactly max_steps optimizer updates (epoch-end logs still
            # run because `break` exits only the inner batch loop).
            if max_steps > 0 and step >= max_steps:
                break
            batch = _to_device(batch, device)

            if micro_idx == 0:
                step += 1
                lr = _cosine_lr(step, total_steps=total_steps, warmup=warmup_steps,
                                base_lr=base_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            if micro_idx == 0:
                optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                step_result = _step(model, batch, mode, focal_gamma, focal_alpha, label_smoothing)
            # Scale so the accumulated gradient equals the mean over the full
            # effective batch, matching a single large-batch step.
            (step_result["loss"] / grad_accum).backward()
            micro_idx += 1
            if micro_idx < grad_accum:
                continue                      # keep accumulating; no optimizer step
            micro_idx = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += float(step_result["loss"].detach().item())
            epoch_steps += 1

            if step % log_every == 0:
                log = {
                    "train/loss": _reduce_mean(float(step_result["loss"].item()), device, distributed, world_size),
                    "train/loss_amp": _reduce_mean(float(step_result["loss_amp"].item()), device, distributed, world_size),
                    "train/loss_morph": _reduce_mean(float(step_result["loss_morph"].item()), device, distributed, world_size),
                    "train/lr": float(lr),
                    "train/epoch": float(epoch),
                    "train/mode": 0.0 if mode == "A" else 1.0,
                }
                if is_main:
                    elapsed = time.time() - epoch_start
                    samples = epoch_steps * per_gpu_bs * world_size
                    tps = samples / max(elapsed, 1e-6)
                    print(f"epoch={epoch} step={step} mode={mode} loss={log['train/loss']:.4f} "
                          f"amp={log['train/loss_amp']:.4f} morph={log['train/loss_morph']:.4f} "
                          f"lr={lr:.2e} tps={tps:.1f}", flush=True)
                    if wb is not None:
                        wb.log(log, step=step)

        epoch_time = time.time() - epoch_start
        epoch_mean_loss = epoch_loss / max(epoch_steps, 1)
        if is_main:
            print(f"[epoch_end] epoch={epoch} mean_loss={epoch_mean_loss:.4f} time={epoch_time:.1f}s steps={epoch_steps}", flush=True)
            if wb is not None:
                wb.log({"epoch/mean_loss": epoch_mean_loss, "epoch/seconds": epoch_time}, step=step)

        # Validation each epoch (or per cadence). Plots overwrite the previous run's plots.
        if val_every_epoch > 0 and ((epoch + 1) % val_every_epoch == 0):
            if is_main:
                save_plots = plot_every_epoch > 0 and ((epoch + 1) % plot_every_epoch == 0)
                val_metrics, val_time, val_samples = _run_eval(
                    model=_base(model), val_loader=val_loader, device=device,
                    autocast_ctx=autocast_ctx,
                    max_batches=int(cfg["validation"].get("max_batches", 100)),
                    focal_gamma=focal_gamma, focal_alpha=focal_alpha, label_smoothing=label_smoothing,
                    plot_dir=(output_dir / "plots"),
                    max_plots=int(cfg["validation"].get("max_plots", 4)),
                    variant=args.variant, save_plots=save_plots,
                )
                tps = val_samples / max(val_time, 1e-6)
                print(f"[val] epoch={epoch} step={step} time={val_time:.1f}s tps={tps:.1f} {val_metrics}", flush=True)
                if wb is not None:
                    wb.log({**val_metrics, "val/seconds": val_time, "val/samples_per_sec": tps}, step=step)
            if distributed:
                dist.barrier()

        if ckpt_every_epoch > 0 and ((epoch + 1) % ckpt_every_epoch == 0):
            if is_main:
                save_checkpoint(
                    path=str(output_dir / f"checkpoint_epoch_{epoch}.pt"),
                    model=_base(model), optimizer=optimizer, scheduler=None,
                    epoch=epoch, step=step, cfg=cfg,
                    extra_state={
                        "batch_idx": steps_per_epoch - 1,
                        "python_random_state": random.getstate(),
                        "numpy_random_state": np.random.get_state(),
                        "torch_random_state": torch.random.get_rng_state(),
                        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                )
                save_checkpoint(
                    path=str(output_dir / "checkpoint_last.pt"),
                    model=_base(model), optimizer=optimizer, scheduler=None,
                    epoch=epoch, step=step, cfg=cfg,
                    extra_state={"batch_idx": steps_per_epoch - 1},
                )
            if distributed:
                dist.barrier()

        resume_batch_idx = 0

    # Final checkpoint + summary JSON.
    if is_main:
        save_checkpoint(
            path=str(output_dir / "checkpoint_last.pt"),
            model=_base(model), optimizer=optimizer, scheduler=None,
            epoch=epochs - 1, step=step, cfg=cfg,
            extra_state={"batch_idx": steps_per_epoch - 1},
        )
        # One last validation pass with plots and persisted JSON summary.
        val_metrics, val_time, val_samples = _run_eval(
            model=_base(model), val_loader=val_loader, device=device,
            autocast_ctx=autocast_ctx,
            max_batches=int(cfg["validation"].get("max_batches", 100)),
            focal_gamma=focal_gamma, focal_alpha=focal_alpha, label_smoothing=label_smoothing,
            plot_dir=(output_dir / "plots"),
            max_plots=int(cfg["validation"].get("max_plots", 4)),
            variant=args.variant, save_plots=True,
        )
        summary = {
            "variant": args.variant, "params_total": total_params, "epochs_run": epochs,
            "final_step": step, "per_gpu_batch_size": per_gpu_bs, "world_size": world_size,
            **val_metrics, "val/seconds": val_time, "val/samples_per_sec": val_samples / max(val_time, 1e-6),
        }
        with open(output_dir / f"long_{args.variant}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[final_val] {val_metrics}", flush=True)
        if wb is not None:
            wb.summary.update(summary)
            wb.finish()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
