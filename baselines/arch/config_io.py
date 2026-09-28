from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(config_path: str) -> dict[str, Any]:
    """Permissive loader for architecture-analysis configs.

    Intentionally does NOT enforce the hub.config 20-epoch budget so this sweep
    can run 50 epochs per the architecture-analysis plan.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a mapping")

    # two backbones:
    #   - "hybrid"  : the original HybridPoseidonTransformer (used by trainers/train_fused.py)
    #   - "seq2seq" : OperatorLearningModel (used by benchmarks/ sweep)
    if "model_type" not in raw:
        raw["model_type"] = "hybrid"
    allowed = {"hybrid", "seq2seq"}
    if str(raw["model_type"]).strip() not in allowed:
        raise ValueError(
            f"config requires model_type in {sorted(allowed)}, got {raw['model_type']!r}"
        )

    # dataset.path is required for file-backed datasets but not for synthetic
    # procedural pretraining (`dataset.synthetic: true`), which generates
    # samples on the fly with no .nc file on disk.
    _ds = raw.get("dataset", {})
    if not bool(_ds.get("synthetic", False)) and not str(_ds.get("path", "")).strip():
        raise ValueError("dataset.path is required (unless dataset.synthetic is true)")

    raw.setdefault("runtime", {})
    raw["runtime"].setdefault("seed", 42)
    raw["runtime"].setdefault("tf32", True)

    raw.setdefault("training", {})
    raw["training"].setdefault("epochs", 50)
    raw["training"].setdefault("lr", 1e-4)
    raw["training"].setdefault("weight_decay", 0.01)
    raw["training"].setdefault("warmup_steps", 2000)
    raw["training"].setdefault("grad_clip", 1.0)
    raw["training"].setdefault("mixed_precision", "bf16")
    raw["training"].setdefault("log_every", 50)
    raw["training"].setdefault("val_every_epoch", 1)
    raw["training"].setdefault("checkpoint_every_epoch", 5)
    raw["training"].setdefault("focal_gamma", 2.0)
    raw["training"].setdefault("focal_alpha", None)
    raw["training"].setdefault("hybrid_mode_cycle", ["A", "B"])
    raw["training"].setdefault("compile", True)
    raw["training"].setdefault("compile_mode", "default")
    raw["training"].setdefault("compile_fullgraph", False)

    raw.setdefault("wandb", {})
    raw["wandb"].setdefault("enabled", True)
    raw["wandb"].setdefault("project", "operator-learning-hub-arch")
    raw["wandb"].setdefault("entity", None)
    raw["wandb"].setdefault("run_name", None)
    raw["wandb"].setdefault("mode", "online")

    raw.setdefault("validation", {})
    raw["validation"].setdefault("max_batches", 16)
    raw["validation"].setdefault("decoder_enabled", False)
    raw["validation"].setdefault("output_dir", os.path.join(os.environ.get("PHAEDRA_OUTPUT_ROOT", "."), "operators", raw.get("_variant", "arch")))

    raw.setdefault("budget", {})
    raw["budget"].setdefault("target_params", None)
    raw["budget"].setdefault("tolerance_frac", 0.15)

    return raw
