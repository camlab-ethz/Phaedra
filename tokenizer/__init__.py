"""Phaedra tokenizer and the three baseline autoencoders (FSQ, VQ-VAE-2, continuous).

Vendored from the research repository with only import lines rewritten, so the
module/attribute names (and therefore checkpoint keys) are unchanged.

`MODEL_REGISTRY` maps the model names used throughout the pipeline to the
training system class and the packaged hyper-parameter config.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class ModelSpec:
    system: str          # "module:Class" inside this package
    config: str          # yaml under tokenizer/configs
    model_type: str      # phaedra | fsq | vqvae2 | continuous
    default_weights: str # sub-directory of $PHAEDRA_OUTPUT_ROOT/tokenizers


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Phaedra_AE_FSQ_4x4": ModelSpec("tokenizer.systems:PhaedraAEFSQSystem", "model_phaedra_ae_fsq_4x4.yaml", "phaedra", "phaedra_4x4"),
    "AE_FSQ":             ModelSpec("tokenizer.systems:FSQAESystem",        "model_ae_fsq.yaml",             "fsq",        "fsq"),
    "AE_VQVAE2":          ModelSpec("tokenizer.systems:VQVAE2AESystem",     "model_ae_vq2.yaml",             "vqvae2",     "vqvae2"),
    "AE_Continuous":      ModelSpec("tokenizer.systems:ContinuousAESystem", "model_ae_continuous.yaml",      "continuous", "continuous"),
}


def config_path(model_name: str) -> Path:
    return CONFIG_DIR / MODEL_REGISTRY[model_name].config


def system_class(model_name: str):
    mod, cls = MODEL_REGISTRY[model_name].system.split(":")
    return getattr(importlib.import_module(mod), cls)


def build_system(model_name: str, config_override: str | Path | None = None):
    """Instantiate the training system (model + optimizer logic) for a model name."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(str(config_override) if config_override else str(config_path(model_name)))
    return system_class(model_name)(cfg)
