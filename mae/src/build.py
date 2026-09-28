"""Build a TokenMAE from a training config (shared by mae/test.py and the release tools).

Vocabulary sizes and the token grid are read from the first token file listed in
the config, exactly as the training script does, and `num_datasets` is the number
of token files (the dataset embedding has one row per file).
"""
from __future__ import annotations

from pathlib import Path

import netCDF4 as nc

from mae.src.model import MAEConfig, TokenMAE


def token_paths(train_cfg: dict) -> list[str]:
    ds = train_cfg["dataset"]
    return [str(p) for p in (ds.get("paths") or [ds["path"]])]


def mae_config_from_train_cfg(train_cfg: dict) -> MAEConfig:
    token_type = str(train_cfg["dataset"]["token_type"])
    paths = token_paths(train_cfg)
    with nc.Dataset(Path(paths[0]), "r") as tok:
        if token_type == "phaedra":
            vocab_amp = int(tok.getncattr("amplitude_codebook_size"))
            vocab_morph = int(tok.getncattr("morphology_codebook_size"))
            vocab_fsq = None
        else:
            vocab_amp = vocab_morph = None
            vocab_fsq = int(tok.getncattr("codebook_size"))
        grid_size = int(tok.dimensions["token_x"].size)
    m, t = train_cfg["model"], train_cfg["training"]
    return MAEConfig(
        token_type=token_type,
        vocab_amp=vocab_amp,
        vocab_morph=vocab_morph,
        vocab_fsq=vocab_fsq,
        num_vars=len(train_cfg["dataset"]["variables"]),
        grid_size=grid_size,
        embed_dim=int(m["embed_dim"]),
        encoder_depth=int(m["encoder_depth"]),
        decoder_depth=int(m["decoder_depth"]),
        num_heads=int(m["num_heads"]),
        mlp_ratio=int(m["mlp_ratio"]),
        mask_ratio=float(t["mask_ratio"]),
        fusion=str(m["fusion"]),
        num_datasets=len(paths),
    )


def build_mae(train_cfg: dict) -> TokenMAE:
    return TokenMAE(mae_config_from_train_cfg(train_cfg))
