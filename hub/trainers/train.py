from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

from hub.config import load_config
from hub.data import build_dataloaders
from hub.eval.metrics import per_variable_relative_l1, relative_l1
from hub.eval.plots import save_sample_overview_plot
from hub.models import build_model
from hub.models.diffusion_process import MaskedDiscreteDiffusion
from hub.utils.checkpoint import load_checkpoint, save_checkpoint
from hub.utils.phaedra_decoder import decode_phaedra_tokens, load_phaedra_decoder
from hub.utils.runtime import configure_torch, get_device, set_seed
from hub.utils.wandb_utils import grad_norm_l2, max_abs_grad, maybe_init_wandb


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
    t = t / float(world_size)
    return float(t.item())


def _base_model(model):
    return model.module if isinstance(model, DDP) else model


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _restore_python_rng_state(state: Any) -> None:
    if state is None:
        return
    random.setstate(state)


def _restore_numpy_rng_state(state: Any) -> None:
    if state is None:
        return
    try:
        np.random.set_state(state)
        return
    except Exception:
        pass

    if isinstance(state, (list, tuple)) and len(state) >= 2:
        alg = str(state[0])
        keys = np.asarray(state[1], dtype=np.uint32)
        pos = int(state[2]) if len(state) > 2 else 0
        has_gauss = int(state[3]) if len(state) > 3 else 0
        cached_gaussian = float(state[4]) if len(state) > 4 else 0.0
        np.random.set_state((alg, keys, pos, has_gauss, cached_gaussian))
        return

    raise TypeError(f"Unsupported numpy RNG state format: {type(state)}")


def _coerce_torch_rng_state(state: Any) -> torch.Tensor:
    if torch.is_tensor(state):
        return state.detach().to(dtype=torch.uint8, device="cpu").contiguous()

    if isinstance(state, np.ndarray):
        return torch.from_numpy(state.astype(np.uint8, copy=False)).contiguous()

    if isinstance(state, (bytes, bytearray)):
        return torch.tensor(list(state), dtype=torch.uint8)

    if isinstance(state, (list, tuple)):
        return torch.as_tensor(state, dtype=torch.uint8).contiguous()

    raise TypeError(f"Unsupported torch RNG state format: {type(state)}")


def _restore_torch_rng_state(state: Any) -> None:
    if state is None:
        return
    torch.random.set_rng_state(_coerce_torch_rng_state(state))


def _restore_cuda_rng_state_all(state: Any) -> None:
    if state is None or not torch.cuda.is_available():
        return

    device_count = int(torch.cuda.device_count())
    if device_count <= 0:
        return
    current_device = int(torch.cuda.current_device())

    if torch.is_tensor(state) or isinstance(state, (np.ndarray, bytes, bytearray)):
        # Single-state checkpoints are common when resuming from different GPU counts.
        torch.cuda.set_rng_state(_coerce_torch_rng_state(state), device=current_device)
        return

    if isinstance(state, (list, tuple)):
        states = [_coerce_torch_rng_state(x) for x in state]
        if not states:
            return
        if len(states) == device_count:
            torch.cuda.set_rng_state_all(states)
        else:
            # Fallback for world-size/device-count changes between save and resume.
            idx = min(current_device, len(states) - 1)
            torch.cuda.set_rng_state(states[idx], device=current_device)
        return

    raise TypeError(f"Unsupported CUDA RNG state format: {type(state)}")


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    if alpha is not None:
        loss = loss * float(alpha)
    return loss.mean()


def _masked_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    )
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    if alpha is not None:
        loss = loss * float(alpha)
    flat_mask = mask.reshape(-1).float()
    denom = flat_mask.sum().clamp_min(1.0)
    return (loss * flat_mask).sum() / denom


def _decode_fields_if_available(
    decoder,
    pred_amp: torch.Tensor,
    pred_morph: torch.Tensor,
    target_amp: torch.Tensor,
    target_morph: torch.Tensor,
    token_type: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if decoder is None or str(token_type).lower().strip() != "phaedra":
        return None

    pred_fields = []
    target_fields = []
    n_vars = int(pred_amp.shape[0])
    for v in range(n_vars):
        pred_v = decode_phaedra_tokens(
            decoder,
            amp_tokens=pred_amp[v].unsqueeze(0),
            morph_tokens=pred_morph[v].unsqueeze(0),
        )[0, 0].detach().float()
        target_v = decode_phaedra_tokens(
            decoder,
            amp_tokens=target_amp[v].unsqueeze(0),
            morph_tokens=target_morph[v].unsqueeze(0),
        )[0, 0].detach().float()
        pred_fields.append(pred_v)
        target_fields.append(target_v)

    return torch.stack(pred_fields, dim=0), torch.stack(target_fields, dim=0)


def _seq2seq_step(
    model,
    batch: dict[str, Any],
    token_type: str,
    train: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
):
    token_type = str(token_type).lower().strip()
    if token_type == "fsq":
        fsq_logits = model(
            input_tokens=batch["input_tokens"],
            input_time_idx=batch["input_time_idx"],
            output_time_idx=batch["output_time_idx"],
            input_var_ids=batch["input_var_ids"],
            output_var_ids=batch["output_var_ids"],
            input_var_mask=batch["input_var_mask"],
            output_var_mask=batch["output_var_mask"],
            problem_type_id=batch["problem_type_id"],
        )

        bsz, out_vars, h, w = batch["output_tokens"].shape
        target_tokens = batch["output_tokens"].view(bsz, -1)

        loss_tokens = _focal_loss(
            logits=fsq_logits.reshape(-1, fsq_logits.shape[-1]),
            targets=target_tokens.reshape(-1),
            gamma=focal_gamma,
            alpha=focal_alpha,
        )
        pred_tokens = fsq_logits.argmax(dim=-1).view(bsz, out_vars, h, w)
        token_acc_tokens = float((pred_tokens == batch["output_tokens"]).float().mean().item())

        return {
            "loss": loss_tokens,
            "loss_tokens": loss_tokens.detach(),
            "pred_tokens": pred_tokens,
            "token_acc_tokens": token_acc_tokens,
        }

    amp_logits, logits_morph = model(
        input_amp=batch["input_amp"],
        input_morph=batch["input_morph"],
        input_time_idx=batch["input_time_idx"],
        output_time_idx=batch["output_time_idx"],
        input_var_ids=batch["input_var_ids"],
        output_var_ids=batch["output_var_ids"],
        input_var_mask=batch["input_var_mask"],
        output_var_mask=batch["output_var_mask"],
        problem_type_id=batch["problem_type_id"],
    )

    bsz, out_vars, h, w = batch["output_amp"].shape
    target_amp = batch["output_amp"].view(bsz, -1)
    target_morph = batch["output_morph"].view(bsz, -1)

    loss_amp = _focal_loss(
        logits=amp_logits.reshape(-1, amp_logits.shape[-1]),
        targets=target_amp.reshape(-1),
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    loss_morph = _focal_loss(
        logits=logits_morph.reshape(-1, logits_morph.shape[-1]),
        targets=target_morph.reshape(-1),
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    loss = loss_amp + loss_morph

    pred_amp = amp_logits.argmax(dim=-1).view(bsz, out_vars, h, w)
    pred_morph = logits_morph.argmax(dim=-1).view(bsz, out_vars, h, w)

    token_acc_amp = float((pred_amp == batch["output_amp"]).float().mean().item())
    token_acc_morph = float((pred_morph == batch["output_morph"]).float().mean().item())

    return {
        "loss": loss,
        "loss_amp": loss_amp.detach(),
        "loss_morph": loss_morph.detach(),
        "pred_amp": pred_amp,
        "pred_morph": pred_morph,
        "token_acc_amp": token_acc_amp,
        "token_acc_morph": token_acc_morph,
    }


def _diffusion_step(
    model,
    diffusion: MaskedDiscreteDiffusion,
    batch: dict[str, Any],
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
):
    corrupted_morph, corrupted_amp, timesteps, supervised_mask = diffusion.forward_process(
        seq_morph=batch["sequence_morph"],
        seq_amp=batch["sequence_amp"],
        target_nonpad_mask=batch["target_nonpad_mask"],
    )

    morph_logits, amp_logits = model(
        morph_tokens=corrupted_morph,
        amp_tokens=corrupted_amp,
        attention_mask=batch["attention_mask"],
        segment_ids=batch["segment_ids"],
        variable_ids=batch["variable_ids"],
        spatial_indices=batch["spatial_indices"],
        lead_time=batch["lead_time"],
        pde_type_id=batch["problem_type_id"],
        diffusion_timestep=timesteps,
    )

    if not supervised_mask.any():
        raise RuntimeError("No supervised diffusion tokens in batch")

    loss_morph = _focal_loss(
        logits=morph_logits[supervised_mask],
        targets=batch["sequence_morph"][supervised_mask],
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    loss_amp = _focal_loss(
        logits=amp_logits[supervised_mask],
        targets=batch["sequence_amp"][supervised_mask],
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    loss = loss_morph + loss_amp

    pred_morph_all = morph_logits.argmax(dim=-1)
    pred_amp_all = amp_logits.argmax(dim=-1)

    token_acc_morph = float((pred_morph_all[supervised_mask] == batch["sequence_morph"][supervised_mask]).float().mean().item())
    token_acc_amp = float((pred_amp_all[supervised_mask] == batch["sequence_amp"][supervised_mask]).float().mean().item())

    bsz = int(batch["sequence_morph"].shape[0])
    start = int(batch["target_start"].item())
    h = int(batch["grid_height"].item())
    w = int(batch["grid_width"].item())
    vars_out = int(batch["target_amp_grid"].shape[1])

    pred_morph_grid = []
    pred_amp_grid = []
    for i in range(bsz):
        flat_m = pred_morph_all[i, start : start + vars_out * h * w]
        flat_a = pred_amp_all[i, start : start + vars_out * h * w]
        pred_morph_grid.append(flat_m.view(vars_out, h, w))
        pred_amp_grid.append(flat_a.view(vars_out, h, w))

    return {
        "loss": loss,
        "loss_amp": loss_amp.detach(),
        "loss_morph": loss_morph.detach(),
        "pred_amp": torch.stack(pred_amp_grid, dim=0),
        "pred_morph": torch.stack(pred_morph_grid, dim=0),
        "token_acc_amp": token_acc_amp,
        "token_acc_morph": token_acc_morph,
    }


def _hybrid_step(
    model,
    batch: dict[str, Any],
    *,
    morph_source: str,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
):
    out = model(
        source_amp=batch["source_amp"],
        source_morph=batch["source_morph"],
        input_time_idx=batch["input_time_idx"],
        output_time_idx=batch["output_time_idx"],
        active_var_mask=batch["active_var_mask"],
        slot_var_ids=batch["slot_var_ids"],
        target_amp=batch["target_amp"] if str(morph_source).lower() == "gt" else None,
        morph_source=str(morph_source),
    )

    bsz, max_vars, h, w = batch["target_amp"].shape
    target_amp = batch["target_amp"].view(bsz, -1)
    target_morph = batch["target_morph"].view(bsz, -1)

    active_mask = batch["active_var_mask"][:, :, None, None].expand_as(batch["target_amp"])
    active_mask_flat = active_mask.reshape(bsz, -1)

    loss_amp = _masked_focal_loss(
        logits=out.logits_amp,
        targets=target_amp,
        mask=active_mask_flat,
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    loss_morph = _masked_focal_loss(
        logits=out.logits_morph,
        targets=target_morph,
        mask=active_mask_flat,
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    loss = loss_amp + loss_morph

    pred_amp = out.pred_amp
    pred_morph = out.pred_morph

    token_acc_amp = float((pred_amp[active_mask] == batch["target_amp"][active_mask]).float().mean().item())
    token_acc_morph = float((pred_morph[active_mask] == batch["target_morph"][active_mask]).float().mean().item())

    return {
        "loss": loss,
        "loss_amp": loss_amp.detach(),
        "loss_morph": loss_morph.detach(),
        "pred_amp": pred_amp,
        "pred_morph": pred_morph,
        "token_acc_amp": token_acc_amp,
        "token_acc_morph": token_acc_morph,
    }


def _run_validation(
    cfg: dict[str, Any],
    model,
    val_loader,
    device: torch.device,
    decoder,
    diffusion: MaskedDiscreteDiffusion | None,
    step: int,
    output_dir: Path,
) -> dict[str, float]:
    model.eval()
    token_type = str(cfg.get("dataset", {}).get("token_type", "phaedra")).lower().strip()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for stale in plot_dir.glob("val_sample*.png"):
        stale.unlink(missing_ok=True)
    for stale in plot_dir.glob("val_tokens_step*_sample*.png"):
        stale.unlink(missing_ok=True)
    (plot_dir / "val_fields_latest.png").unlink(missing_ok=True)

    if token_type == "fsq":
        metrics = {
            "val/loss": 0.0,
            "val/loss_tokens": 0.0,
            "val/token_acc_tokens": 0.0,
        }
    else:
        metrics = {
            "val/loss": 0.0,
            "val/loss_amp": 0.0,
            "val/loss_morph": 0.0,
            "val/token_acc_amp": 0.0,
            "val/token_acc_morph": 0.0,
        }
    count = 0
    recon_count = 0
    plots_written = 0
    max_plots = int(cfg["validation"]["max_plots"])

    focal_gamma = float(cfg["training"].get("focal_gamma", 2.0))
    focal_alpha = cfg["training"].get("focal_alpha", None)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= int(cfg["validation"]["max_batches"]):
                break
            batch = _to_device(batch, device)

            if cfg["model_type"] == "seq2seq":
                out = _seq2seq_step(
                    model,
                    batch,
                    token_type=token_type,
                    train=False,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
                if token_type == "fsq":
                    target_tokens = batch["output_tokens"]
                else:
                    target_amp = batch["output_amp"]
                    target_morph = batch["output_morph"]
            elif cfg["model_type"] == "diffusion":
                assert diffusion is not None
                out = _diffusion_step(
                    model,
                    diffusion,
                    batch,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
                target_amp = batch["target_amp_grid"]
                target_morph = batch["target_morph_grid"]
            else:
                out = _hybrid_step(
                    model,
                    batch,
                    morph_source="pred",
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
                target_amp = batch["target_amp"]
                target_morph = batch["target_morph"]

            metrics["val/loss"] += float(out["loss"].item())
            if cfg["model_type"] == "seq2seq" and token_type == "fsq":
                metrics["val/loss_tokens"] += float(out["loss_tokens"].item())
                metrics["val/token_acc_tokens"] += float(out["token_acc_tokens"])
            else:
                metrics["val/loss_amp"] += float(out["loss_amp"].item())
                metrics["val/loss_morph"] += float(out["loss_morph"].item())
                metrics["val/token_acc_amp"] += float(out["token_acc_amp"])
                metrics["val/token_acc_morph"] += float(out["token_acc_morph"])
            count += 1

            if token_type != "fsq":
                bsz = int(target_amp.shape[0])
                for sample_idx in range(bsz):
                    dec = _decode_fields_if_available(
                        decoder,
                        pred_amp=out["pred_amp"][sample_idx],
                        pred_morph=out["pred_morph"][sample_idx],
                        target_amp=target_amp[sample_idx],
                        target_morph=target_morph[sample_idx],
                        token_type=token_type,
                    )
                    if dec is None:
                        continue

                    pred_field, target_field = dec
                    names = list(batch["var_names"][sample_idx])
                    recon_count += 1

                    metrics["val/recon_rel_l1"] = metrics.get("val/recon_rel_l1", 0.0) + relative_l1(pred_field, target_field)
                    for var, val in per_variable_relative_l1(pred_field, target_field, names).items():
                        key = f"val/recon_rel_l1_{var}"
                        metrics[key] = metrics.get(key, 0.0) + float(val)

                    if plots_written < max_plots:
                        save_sample_overview_plot(
                            path=str(plot_dir / f"val_sample{plots_written}.png"),
                            target_morph=target_morph[sample_idx],
                            pred_morph=out["pred_morph"][sample_idx],
                            target_amp=target_amp[sample_idx],
                            pred_amp=out["pred_amp"][sample_idx],
                            target_field=target_field,
                            pred_field=pred_field,
                            var_names=names,
                        )
                        plots_written += 1

    if count > 0:
        if token_type == "fsq":
            for key in ["val/loss", "val/loss_tokens", "val/token_acc_tokens"]:
                metrics[key] = float(metrics[key] / count)
        else:
            for key in ["val/loss", "val/loss_amp", "val/loss_morph", "val/token_acc_amp", "val/token_acc_morph"]:
                metrics[key] = float(metrics[key] / count)

    if recon_count > 0:
        if "val/recon_rel_l1" in metrics:
            metrics["val/recon_rel_l1"] = float(metrics["val/recon_rel_l1"] / recon_count)
        for key in list(metrics.keys()):
            if key.startswith("val/recon_rel_l1_"):
                metrics[key] = float(metrics[key] / recon_count)

    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified trainer for operator-learning hub")
    parser.add_argument("--config", "--configs", dest="config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    token_type = str(cfg.get("dataset", {}).get("token_type", "phaedra")).lower().strip()

    set_seed(int(cfg["runtime"]["seed"]))
    configure_torch(tf32=bool(cfg["runtime"].get("tf32", True)))
    device = get_device()
    distributed, rank, world_size, _local_rank, is_main = _dist_info()

    device_name = str(device)
    if device.type == "cuda":
        device_name = f"cuda:{torch.cuda.current_device()}"
    print(
        "[startup] "
        f"rank={rank}/{world_size} "
        f"local_rank={_local_rank} "
        f"pid={os.getpid()} "
        f"host={os.uname().nodename} "
        f"device={device_name}",
        flush=True,
    )

    train_loader, val_loader, train_sampler = build_dataloaders(
        cfg,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    model = build_model(cfg).to(device)
    if distributed:
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)
        else:
            model = DDP(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(
            "[train] "
            f"model_type={cfg['model_type']} "
            f"world_size={world_size} "
            f"params_total={total_params} ({total_params / 1e6:.2f}M) "
            f"params_trainable={trainable_params} ({trainable_params / 1e6:.2f}M)"
        )
        if cfg["model_type"] == "hybrid":
            base_model = _base_model(model)
            amp_params = sum(p.numel() for p in base_model.amp_model.parameters())
            morph_params = sum(p.numel() for p in base_model.morph_model.parameters())
            print(
                "[train] "
                f"hybrid_amp_params={amp_params} ({amp_params / 1e6:.2f}M) "
                f"hybrid_morph_params={morph_params} ({morph_params / 1e6:.2f}M)"
            )

    wb = maybe_init_wandb(cfg) if is_main else None

    decoder = None
    if is_main and bool(cfg["validation"].get("decoder_enabled", False)):
        if token_type != "phaedra":
            print("[validation] decoder_enabled ignored for non-phaedra token_type")
        else:
            decoder = load_phaedra_decoder(
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

    optimizer = None
    optimizer_amp = None
    optimizer_morph = None
    if cfg["model_type"] == "hybrid":
        base_model = _base_model(model)
        amp_lr_raw = cfg["training"].get("hybrid_lr_amp")
        morph_lr_raw = cfg["training"].get("hybrid_lr_morph")
        amp_wd_raw = cfg["training"].get("hybrid_weight_decay_amp")
        morph_wd_raw = cfg["training"].get("hybrid_weight_decay_morph")
        amp_lr = float(amp_lr_raw) if amp_lr_raw is not None else float(cfg["training"]["lr"])
        morph_lr = float(morph_lr_raw) if morph_lr_raw is not None else float(cfg["training"]["lr"])
        amp_wd = float(amp_wd_raw) if amp_wd_raw is not None else float(cfg["training"]["weight_decay"])
        morph_wd = float(morph_wd_raw) if morph_wd_raw is not None else float(cfg["training"]["weight_decay"])
        optimizer_amp = AdamW(
            base_model.amp_model.parameters(),
            lr=amp_lr,
            weight_decay=amp_wd,
            betas=(0.9, 0.95),
        )
        optimizer_morph = AdamW(
            base_model.morph_model.parameters(),
            lr=morph_lr,
            weight_decay=morph_wd,
            betas=(0.9, 0.95),
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            betas=(0.9, 0.95),
        )

    epochs = int(cfg["training"]["epochs"])
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)

    if is_main:
        per_gpu_train_bs = int(cfg["dataset"]["batch_size_train"])
        per_gpu_val_bs = int(cfg["dataset"]["batch_size_val"])
        print(
            "[data] "
            f"train_samples={train_samples} "
            f"val_samples={val_samples} "
            f"steps_per_epoch={steps_per_epoch} "
            f"per_gpu_batch_train={per_gpu_train_bs} "
            f"global_batch_train={per_gpu_train_bs * world_size} "
            f"per_gpu_batch_val={per_gpu_val_bs} "
            f"global_batch_val={per_gpu_val_bs * world_size}",
            flush=True,
        )

    diffusion = None
    if cfg["model_type"] == "diffusion":
        amp_mask_id = int(cfg["dataset"]["amp_vocab_size"]) + 2
        morph_mask_id = int(cfg["dataset"]["morph_vocab_size"]) + 2
        diffusion = MaskedDiscreteDiffusion(
            num_steps=int(cfg["model"].get("diffusion_steps", 32)),
            morph_mask_id=morph_mask_id,
            amp_mask_id=amp_mask_id,
        )

    output_dir = Path(cfg["validation"]["output_dir"]).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    warmup_steps = int(cfg["training"]["warmup_steps"])
    focal_gamma = float(cfg["training"].get("focal_gamma", 2.0))
    focal_alpha = cfg["training"].get("focal_alpha", None)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)

    log_every = int(cfg["training"]["log_every"])
    val_every = int(cfg["training"]["val_every"])
    checkpoint_every = int(cfg["training"]["checkpoint_every"])

    step = 0
    start_epoch = 0
    resume_batch_idx = 0
    resume_exact_batch = bool(cfg["training"].get("resume_exact_batch", True))

    resume_path = args.resume or cfg["training"].get("resume_from")
    if resume_path:
        state = load_checkpoint(str(resume_path), device=device)
        _base_model(model).load_state_dict(state["model"], strict=True)
        if "optimizer" in state and state["optimizer"] is not None:
            if cfg["model_type"] == "hybrid":
                opt_state = state["optimizer"]
                if isinstance(opt_state, dict):
                    if optimizer_amp is not None and opt_state.get("amp") is not None:
                        optimizer_amp.load_state_dict(opt_state["amp"])
                    if optimizer_morph is not None and opt_state.get("morph") is not None:
                        optimizer_morph.load_state_dict(opt_state["morph"])
                else:
                    if is_main:
                        print("[resume] warning: hybrid checkpoint optimizer state is not a dict; skipping")
            else:
                optimizer.load_state_dict(state["optimizer"])

        step = int(state.get("step", 0))
        saved_epoch = int(state.get("epoch", 0))
        saved_batch_idx = int(state.get("batch_idx", -1))

        if saved_batch_idx + 1 >= steps_per_epoch:
            start_epoch = saved_epoch + 1
            resume_batch_idx = 0
        else:
            if resume_exact_batch:
                start_epoch = saved_epoch
                resume_batch_idx = saved_batch_idx + 1
            else:
                start_epoch = saved_epoch + 1
                resume_batch_idx = 0

        py_state = state.get("python_random_state")
        np_state = state.get("numpy_random_state")
        torch_state = state.get("torch_random_state")
        cuda_state = state.get("cuda_random_state_all")

        try:
            _restore_python_rng_state(py_state)
        except Exception as exc:
            if is_main:
                print(f"[resume] warning: could not restore python RNG state: {exc}")

        try:
            _restore_numpy_rng_state(np_state)
        except Exception as exc:
            if is_main:
                print(f"[resume] warning: could not restore numpy RNG state: {exc}")

        try:
            _restore_torch_rng_state(torch_state)
        except Exception as exc:
            if is_main:
                print(f"[resume] warning: could not restore torch RNG state: {exc}")

        try:
            _restore_cuda_rng_state_all(cuda_state)
        except Exception as exc:
            if is_main:
                print(f"[resume] warning: could not restore CUDA RNG state: {exc}")

        if is_main:
            print(
                f"[resume] checkpoint={resume_path} step={step} "
                f"start_epoch={start_epoch} resume_batch_idx={resume_batch_idx} "
                f"resume_exact_batch={resume_exact_batch}"
            )
            if not resume_exact_batch and saved_batch_idx + 1 < steps_per_epoch:
                print(
                    "[resume] info: skipping exact mid-epoch replay and resuming at next epoch boundary "
                    f"(saved_epoch={saved_epoch}, saved_batch_idx={saved_batch_idx})"
                )

    if start_epoch >= epochs:
        if is_main:
            print(f"[train] checkpoint already reached epochs={epochs}; nothing to run")
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return


    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < resume_batch_idx:
                if is_main and (
                    batch_idx == 0
                    or (batch_idx + 1) % 200 == 0
                    or (batch_idx + 1) == resume_batch_idx
                ):
                    print(
                        f"[resume] fast-forwarding epoch={epoch} "
                        f"batch={batch_idx + 1}/{resume_batch_idx}"
                    )
                continue

            step += 1
            batch = _to_device(batch, device)

            if cfg["model_type"] == "hybrid":
                amp_base_lr_raw = cfg["training"].get("hybrid_lr_amp")
                morph_base_lr_raw = cfg["training"].get("hybrid_lr_morph")
                amp_base_lr = float(amp_base_lr_raw) if amp_base_lr_raw is not None else float(cfg["training"]["lr"])
                morph_base_lr = float(morph_base_lr_raw) if morph_base_lr_raw is not None else float(cfg["training"]["lr"])
                lr_amp = _cosine_lr(step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=amp_base_lr)
                lr_morph = _cosine_lr(step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=morph_base_lr)
                for group in optimizer_amp.param_groups:
                    group["lr"] = lr_amp
                for group in optimizer_morph.param_groups:
                    group["lr"] = lr_morph
                optimizer_amp.zero_grad(set_to_none=True)
                optimizer_morph.zero_grad(set_to_none=True)
                lr = lr_amp
            else:
                lr = _cosine_lr(step, total_steps=total_steps, warmup_steps=warmup_steps, base_lr=float(cfg["training"]["lr"]))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)

            if cfg["model_type"] == "seq2seq":
                out = _seq2seq_step(
                    model,
                    batch,
                    token_type=token_type,
                    train=True,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
            elif cfg["model_type"] == "diffusion":
                assert diffusion is not None
                out = _diffusion_step(
                    model,
                    diffusion,
                    batch,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
            else:
                morph_source = str(cfg["training"].get("hybrid_morph_input", "gt"))
                out = _hybrid_step(
                    model,
                    batch,
                    morph_source=morph_source,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )

            out["loss"].backward()
            grad_pre = grad_norm_l2(model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["training"]["grad_clip"]))
            grad_post = grad_norm_l2(model.parameters())
            grad_abs = max_abs_grad(model.parameters())
            if cfg["model_type"] == "hybrid":
                optimizer_amp.step()
                optimizer_morph.step()
            else:
                optimizer.step()

            if step % log_every == 0:
                train_log = {
                    "train/loss": _reduce_mean(float(out["loss"].item()), device, distributed, world_size),
                    "train/lr": float(lr),
                    "train/grad_norm_pre_clip": _reduce_mean(float(grad_pre), device, distributed, world_size),
                    "train/grad_norm_post_clip": _reduce_mean(float(grad_post), device, distributed, world_size),
                    "train/grad_max_abs": _reduce_mean(float(grad_abs), device, distributed, world_size),
                }
                if cfg["model_type"] == "seq2seq" and token_type == "fsq":
                    train_log["train/loss_tokens"] = _reduce_mean(
                        float(out["loss_tokens"].item()), device, distributed, world_size
                    )
                    train_log["train/token_acc_tokens"] = _reduce_mean(
                        float(out["token_acc_tokens"]), device, distributed, world_size
                    )
                else:
                    train_log["train/loss_amp"] = _reduce_mean(float(out["loss_amp"].item()), device, distributed, world_size)
                    train_log["train/loss_morph"] = _reduce_mean(float(out["loss_morph"].item()), device, distributed, world_size)
                    train_log["train/token_acc_amp"] = _reduce_mean(float(out["token_acc_amp"]), device, distributed, world_size)
                    train_log["train/token_acc_morph"] = _reduce_mean(float(out["token_acc_morph"]), device, distributed, world_size)
                if cfg["model_type"] == "hybrid":
                    train_log["train/lr_amp"] = float(lr_amp)
                    train_log["train/lr_morph"] = float(lr_morph)
                if is_main:
                    if cfg["model_type"] == "seq2seq" and token_type == "fsq":
                        print(
                            f"epoch={epoch} step={step} loss={train_log['train/loss']:.4f} "
                            f"tokens={train_log['train/loss_tokens']:.4f} "
                            f"grad_pre={train_log['train/grad_norm_pre_clip']:.3f} "
                            f"grad_post={train_log['train/grad_norm_post_clip']:.3f}"
                        )
                    else:
                        print(
                            f"epoch={epoch} step={step} loss={train_log['train/loss']:.4f} "
                            f"amp={train_log['train/loss_amp']:.4f} morph={train_log['train/loss_morph']:.4f} "
                            f"grad_pre={train_log['train/grad_norm_pre_clip']:.3f} "
                            f"grad_post={train_log['train/grad_norm_post_clip']:.3f}"
                        )
                    if wb is not None:
                        wb.log(train_log, step=step)

            if val_every > 0 and step % val_every == 0:
                if is_main:
                    val_metrics = _run_validation(
                        cfg=cfg,
                        model=_base_model(model),
                        val_loader=val_loader,
                        device=device,
                        decoder=decoder,
                        diffusion=diffusion,
                        step=step,
                        output_dir=output_dir,
                    )
                    print(f"[val] step={step} {val_metrics}")
                    if wb is not None:
                        wb.log(val_metrics, step=step)
                if distributed:
                    dist.barrier()

            if checkpoint_every > 0 and step % checkpoint_every == 0:
                if is_main:
                    optimizer_payload = (
                        {"amp": optimizer_amp, "morph": optimizer_morph}
                        if cfg["model_type"] == "hybrid"
                        else optimizer
                    )
                    save_checkpoint(
                        path=str(output_dir / f"checkpoint_step_{step}.pt"),
                        model=_base_model(model),
                        optimizer=optimizer_payload,
                        scheduler=None,
                        epoch=epoch,
                        step=step,
                        cfg=cfg,
                        extra_state={
                            "batch_idx": int(batch_idx),
                            "python_random_state": random.getstate(),
                            "numpy_random_state": np.random.get_state(),
                            "torch_random_state": torch.random.get_rng_state(),
                            "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                    )
                    save_checkpoint(
                        path=str(output_dir / "checkpoint_last.pt"),
                        model=_base_model(model),
                        optimizer=optimizer_payload,
                        scheduler=None,
                        epoch=epoch,
                        step=step,
                        cfg=cfg,
                        extra_state={
                            "batch_idx": int(batch_idx),
                            "python_random_state": random.getstate(),
                            "numpy_random_state": np.random.get_state(),
                            "torch_random_state": torch.random.get_rng_state(),
                            "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                    )
                if distributed:
                    dist.barrier()

    if is_main:
        optimizer_payload = (
            {"amp": optimizer_amp, "morph": optimizer_morph}
            if cfg["model_type"] == "hybrid"
            else optimizer
        )
        save_checkpoint(
            path=str(output_dir / "checkpoint_last.pt"),
            model=_base_model(model),
            optimizer=optimizer_payload,
            scheduler=None,
            epoch=epochs - 1,
            step=step,
            cfg=cfg,
            extra_state={
                "batch_idx": int(steps_per_epoch - 1),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.random.get_rng_state(),
                "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        )

        if wb is not None:
            wb.finish()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
