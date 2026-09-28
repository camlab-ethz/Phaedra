from __future__ import annotations

import torch


def relative_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    num = torch.mean(torch.abs(pred - target))
    den = torch.mean(torch.abs(target)).clamp_min(1e-8)
    return float((num / den).item())


def per_variable_relative_l1(pred: torch.Tensor, target: torch.Tensor, var_names: list[str]) -> dict[str, float]:
    # pred/target: [V, H, W]
    out: dict[str, float] = {}
    for i, name in enumerate(var_names):
        out[str(name)] = relative_l1(pred[i], target[i])
    return out
