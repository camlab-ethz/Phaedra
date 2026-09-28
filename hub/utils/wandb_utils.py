from __future__ import annotations

from typing import Any

import torch


def maybe_init_wandb(cfg: dict[str, Any]):
    wb = cfg["wandb"]
    if not bool(wb.get("enabled", False)):
        return None
    try:
        import wandb

        return wandb.init(
            project=str(wb["project"]),
            entity=wb.get("entity"),
            name=wb.get("run_name"),
            config=cfg,
            mode=str(wb.get("mode", "online")),
        )
    except Exception as exc:  # pragma: no cover
        print(f"[warn] wandb init failed; continuing without wandb: {exc}")
        return None


def grad_norm_l2(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(torch.sum(p.grad.detach().float() ** 2).item())
    return float(total ** 0.5)


def max_abs_grad(parameters) -> float:
    max_val = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        v = float(p.grad.detach().abs().max().item())
        if v > max_val:
            max_val = v
    return max_val
