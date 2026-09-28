from __future__ import annotations

from typing import Any

import torch.nn as nn

from .diffusion_model import DiffusionTransformerConfig, MaskedPDEDiffusionTransformer
from .hybrid_model import HybridTwoStageModel, build_hybrid_two_stage_config
from .seq2seq_model import OperatorConfig, OperatorLearningModel


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_param_budget(model: nn.Module, budget_cfg: dict[str, Any]) -> None:
    target = int(budget_cfg["target_params"])
    tol = int(budget_cfg["tolerance"])
    total = count_parameters(model)
    lo = target - tol
    hi = target + tol
    if not (lo <= total <= hi):
        raise ValueError(
            f"Model parameter count {total} is outside required budget [{lo}, {hi}]"
        )


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_type = str(cfg["model_type"])
    model_cfg = cfg["model"]
    ds_cfg = cfg["dataset"]
    token_type = str(ds_cfg.get("token_type", "phaedra")).lower().strip()

    if model_type == "seq2seq":
        legacy_decoder_depth = int(model_cfg.get("decoder_depth", 3))
        if token_type == "fsq":
            decoder_depth = int(model_cfg.get("decoder_depth", model_cfg.get("amp_decoder_depth", legacy_decoder_depth)))
            op_cfg = OperatorConfig(
                token_type="fsq",
                vocab_amp=None,
                vocab_morph=None,
                vocab_fsq=int(ds_cfg.get("fsq_vocab_size", ds_cfg.get("morph_vocab_size", 0))),
                num_in_vars=len(ds_cfg["input_variables"]),
                num_out_vars=len(ds_cfg["output_variables"]),
                grid_size=int(model_cfg.get("grid_size", 128)),
                embed_dim=int(model_cfg.get("embed_dim", 256)),
                encoder_depth=int(model_cfg.get("encoder_depth", 18)),
                amp_decoder_depth=decoder_depth,
                morph_decoder_depth=0,
                num_heads=int(model_cfg.get("num_heads", 8)),
                mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
                fusion=str(model_cfg.get("fusion", "none")),
                layerscale_init=float(model_cfg.get("layerscale_init", 1e-5)),
                max_time_index=int(model_cfg.get("max_time_index", 14)),
                problem_type_embed_scale=float(model_cfg.get("problem_type_embed_scale", 1.0)),
                attention_mode=str(model_cfg.get("attention_mode", "full")),
            )
        else:
            op_cfg = OperatorConfig(
                token_type="phaedra",
                vocab_amp=int(ds_cfg["amp_vocab_size"]),
                vocab_morph=int(ds_cfg["morph_vocab_size"]),
                vocab_fsq=None,
                num_in_vars=len(ds_cfg["input_variables"]),
                num_out_vars=len(ds_cfg["output_variables"]),
                grid_size=int(model_cfg.get("grid_size", 128)),
                embed_dim=int(model_cfg.get("embed_dim", 256)),
                encoder_depth=int(model_cfg.get("encoder_depth", 18)),
                amp_decoder_depth=int(model_cfg.get("amp_decoder_depth", legacy_decoder_depth)),
                morph_decoder_depth=int(model_cfg.get("morph_decoder_depth", legacy_decoder_depth)),
                num_heads=int(model_cfg.get("num_heads", 8)),
                mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
                fusion=str(model_cfg.get("fusion", "concat")),
                layerscale_init=float(model_cfg.get("layerscale_init", 1e-5)),
                max_time_index=int(model_cfg.get("max_time_index", 14)),
                problem_type_embed_scale=float(model_cfg.get("problem_type_embed_scale", 1.0)),
                attention_mode=str(model_cfg.get("attention_mode", "full")),
            )
        model = OperatorLearningModel(op_cfg)
    elif model_type == "diffusion":
        if token_type != "phaedra":
            raise ValueError("Diffusion model currently supports token_type=phaedra only")
        tr_cfg = DiffusionTransformerConfig(
            morph_vocab_size=int(ds_cfg["morph_vocab_size"]) + 3,
            amp_vocab_size=int(ds_cfg["amp_vocab_size"]) + 3,
            num_variable_types=len(ds_cfg["variable_order"]),
            variable_pad_id=len(ds_cfg["variable_order"]),
            grid_height=int(model_cfg.get("grid_height", 128)),
            grid_width=int(model_cfg.get("grid_width", 128)),
            max_lead_time=int(model_cfg.get("max_lead_time", 14)),
            diffusion_steps=int(model_cfg.get("diffusion_steps", 32)),
            hidden_dim=int(model_cfg["hidden_dim"]),
            depth=int(model_cfg["depth"]),
            num_heads=int(model_cfg["num_heads"]),
            mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        model = MaskedPDEDiffusionTransformer(tr_cfg)
    elif model_type == "hybrid":
        if token_type != "phaedra":
            raise ValueError("Hybrid model currently supports token_type=phaedra only")
        model = HybridTwoStageModel(build_hybrid_two_stage_config(model_cfg, ds_cfg))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    _assert_param_budget(model, cfg["budget"])
    return model
