"""Trainer for the continuous-latent operator (baseline of the main comparison).

The recipe mirrors the Phaedra / FSQ token transformers: AdamW(0.9, 0.95), wd 0.01,
cosine LR + warmup 1000, bf16 autocast, grad clip 1.0, batch 32/GPU, all2all pairs,
seed 42, 100 epochs. Loss: L1 in latent space. DDP via torchrun. A provenance row
(run, config hash, GPU, wall-clock) is appended to $PHAEDRA_OUTPUT_ROOT/logs/gpu_hours.csv
on completion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hub.utils.checkpoint import load_checkpoint, save_checkpoint
from hub.utils.runtime import configure_torch, get_device, set_seed
from hub.utils.wandb_utils import maybe_init_wandb
from baselines.data_continuous import ContinuousLatentDataConfig, ContinuousLatentDataset
from baselines.models_continuous import build_continuous_operator

REPO = Path(__file__).resolve().parents[1]
GPU_HOURS_CSV = Path(os.environ.get("PHAEDRA_OUTPUT_ROOT", ".")) / "logs" / "gpu_hours.csv"


def _dist_info():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    lr_ = int(os.environ.get("LOCAL_RANK", "0"))
    if ws <= 1:
        return False, 0, 1, lr_, True
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                init_method="env://")
    return True, dist.get_rank(), dist.get_world_size(), lr_, dist.get_rank() == 0


def _cosine_lr(step, total, warmup, base):
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))


def _base(m):
    return m.module if isinstance(m, DDP) else m


@torch.no_grad()
def _validate(model, loader, device, autocast_ctx, max_batches):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        with autocast_ctx():
            pred = model(b["input_latents"].to(device),
                         b["input_time_idx"].to(device),
                         b["output_time_idx"].to(device))
        tot += float(F.l1_loss(pred, b["target_latents"].to(device)).item())
        n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    set_seed(int(cfg["runtime"].get("seed", 42)))
    configure_torch(tf32=True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    distributed, rank, world, local_rank, is_main = _dist_info()
    device = get_device()
    ds_cfg = cfg["dataset"]

    train_ds = ContinuousLatentDataset(ContinuousLatentDataConfig(
        path=ds_cfg["path"], split="train", pair_mode="all_even_forward",
        max_members=ds_cfg.get("max_train_members")))
    val_ds = ContinuousLatentDataset(ContinuousLatentDataConfig(
        path=ds_cfg["path"], split="val", pair_mode="fixed",
        fixed_input_time=0, fixed_output_time=14))

    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                  drop_last=True, seed=int(cfg["runtime"].get("seed", 42)))
               if distributed else None)
    nw = int(ds_cfg.get("num_workers", 8))
    train_loader = DataLoader(train_ds, batch_size=int(ds_cfg["batch_size_train"]),
                              shuffle=(sampler is None), sampler=sampler, num_workers=nw,
                              pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0,
                              prefetch_factor=4 if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=int(ds_cfg.get("batch_size_val", 16)),
                            shuffle=False, num_workers=max(1, nw // 2), pin_memory=True)

    model = build_continuous_operator(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"[train] params={n_params:,} ({n_params / 1e6:.2f}M) "
              f"train={len(train_ds)} val={len(val_ds)} world={world}", flush=True)
        tgt = cfg.get("budget", {}).get("target_params")
        if tgt:
            tol = float(cfg["budget"].get("tolerance_frac", 0.05))
            assert abs(n_params - tgt) <= tgt * tol, \
                f"param budget: {n_params:,} vs target {tgt:,} ±{tol * 100:.0f}%"

    do_compile = bool(cfg["training"].get("compile", True))
    if do_compile:
        model = torch.compile(model, fullgraph=False)
    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index,
                    gradient_as_bucket_view=True, static_graph=True)

    opt = AdamW(model.parameters(), lr=float(cfg["training"]["lr"]),
                weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
                betas=(0.9, 0.95), fused=torch.cuda.is_available())

    epochs = int(cfg["training"]["epochs"])
    spe = len(train_loader)
    total_steps = epochs * spe
    warmup = int(cfg["training"].get("warmup_steps", 1000))
    base_lr = float(cfg["training"]["lr"])
    clip = float(cfg["training"].get("grad_clip", 1.0))
    log_every = int(cfg["training"].get("log_every", 100))
    ckpt_every = int(cfg["training"].get("checkpoint_every_epoch", 10))
    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    def autocast_ctx():
        return (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext())

    wb = maybe_init_wandb(cfg) if is_main else None
    step, start_epoch = 0, 0
    if args.resume:
        st = load_checkpoint(str(args.resume), device=device)
        _base(model).load_state_dict(st["model"], strict=True)
        if st.get("optimizer"):
            opt.load_state_dict(st["optimizer"])
        step = int(st.get("step", 0))
        start_epoch = int(st.get("epoch", 0)) + 1
        if is_main:
            print(f"[resume] step={step} epoch={start_epoch}", flush=True)

    t_start = time.time()
    model.train()
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ep_t0 = time.time()
        for batch in train_loader:
            step += 1
            lr = _cosine_lr(step, total_steps, warmup, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                pred = model(batch["input_latents"].to(device, non_blocking=True),
                             batch["input_time_idx"].to(device),
                             batch["output_time_idx"].to(device))
                loss = F.l1_loss(pred, batch["target_latents"].to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if step % log_every == 0 and is_main:
                print(f"epoch={epoch} step={step} latent_l1={loss.item():.5f} lr={lr:.2e}",
                      flush=True)
                if wb:
                    wb.log({"train/latent_l1": float(loss.item()), "train/lr": lr}, step=step)

        if is_main:
            val_l1 = _validate(_base(model), val_loader, device, autocast_ctx,
                               int(cfg.get("validation", {}).get("max_batches", 16)))
            print(f"[val] epoch={epoch} latent_l1={val_l1:.5f} "
                  f"({time.time() - ep_t0:.0f}s/epoch)", flush=True)
            if wb:
                wb.log({"val/latent_l1": val_l1}, step=step)
            if ckpt_every > 0 and (epoch + 1) % ckpt_every == 0:
                save_checkpoint(path=str(out_dir / f"checkpoint_epoch_{epoch}.pt"),
                                model=_base(model), optimizer=opt, scheduler=None,
                                epoch=epoch, step=step, cfg=cfg)
                save_checkpoint(path=str(out_dir / "checkpoint_last.pt"),
                                model=_base(model), optimizer=opt, scheduler=None,
                                epoch=epoch, step=step, cfg=cfg)
        if distributed:
            dist.barrier()

    if is_main:
        save_checkpoint(path=str(out_dir / "checkpoint_last.pt"), model=_base(model),
                        optimizer=opt, scheduler=None, epoch=epochs - 1, step=step, cfg=cfg)
        wall_h = (time.time() - t_start) / 3600.0
        cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str)
                                  .encode()).hexdigest()[:12]
        GPU_HOURS_CSV.parent.mkdir(parents=True, exist_ok=True)
        new = not GPU_HOURS_CSV.exists()
        with GPU_HOURS_CSV.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["run", "config_sha", "gpu", "n_gpus", "wallclock_h", "gpu_hours"])
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            w.writerow([cfg.get("run_name", out_dir.name), cfg_hash, gpu, world,
                        f"{wall_h:.2f}", f"{wall_h * world:.2f}"])
        print(f"[done] {wall_h:.2f}h wall, {wall_h * world:.2f} GPU-h", flush=True)
        if wb:
            wb.finish()
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
