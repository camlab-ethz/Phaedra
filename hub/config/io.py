from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .schema import HubConfig


def _merge_dataclass_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = asdict(HubConfig())
    merged = OmegaConf.merge(defaults, raw)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping")

    cfg = _merge_dataclass_defaults(raw)
    if str(cfg.get("model_type", "")).strip() not in {"seq2seq", "seq3d", "seq3d_maskgit", "diffusion", "hybrid"}:
        raise ValueError("model_type must be one of: seq2seq, seq3d, seq3d_maskgit, diffusion, hybrid")

    dataset_path = str(cfg.get("dataset", {}).get("path", "")).strip()
    if not dataset_path:
        raise ValueError("dataset.path is required and cannot be empty")

    epochs = int(cfg["training"]["epochs"])
    # if epochs != 20:
    #     raise ValueError(f"This hub enforces a strict 20-epoch budget, got epochs={epochs}")

    return cfg
