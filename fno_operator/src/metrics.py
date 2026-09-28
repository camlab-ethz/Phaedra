from __future__ import annotations

from typing import Any

import numpy as np


def relative_l1(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.mean(np.abs(pred - target)))
    den = float(np.mean(np.abs(target)) + 1.0e-8)
    return num / den


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.sqrt(np.mean((pred - target) ** 2)))
    den = float(np.sqrt(np.mean(target ** 2)) + 1.0e-8)
    return num / den


def wasserstein_1d(pred: np.ndarray, target: np.ndarray) -> float:
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


def default_metric_store(variables: list[str]) -> dict[str, dict[str, list[float]]]:
    return {
        "relative_l1": {name: [] for name in variables},
        "relative_l2": {name: [] for name in variables},
        "w1": {name: [] for name in variables},
    }


def append_metrics(
    metric_store: dict[str, dict[str, list[float]]],
    variables: list[str],
    pred_map: dict[str, np.ndarray],
    target_map: dict[str, np.ndarray],
) -> None:
    for name in variables:
        pred_arr = pred_map[name]
        target_arr = target_map[name]
        metric_store["relative_l1"][name].append(relative_l1(pred_arr, target_arr))
        metric_store["relative_l2"][name].append(relative_l2(pred_arr, target_arr))
        metric_store["w1"][name].append(wasserstein_1d(pred_arr, target_arr))


def summarize_metric_store(metric_store: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
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
