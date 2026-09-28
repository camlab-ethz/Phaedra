from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .seq2seq_model import OperatorConfig, OperatorLearningModel


@dataclass
class HybridTwoStageConfig:
    amp: OperatorConfig
    morph: OperatorConfig
    morph_lead_time_mode: str = "zero"


@dataclass
class HybridTwoStageOutput:
    logits_amp: torch.Tensor
    logits_morph: torch.Tensor
    pred_amp: torch.Tensor
    pred_morph: torch.Tensor


class HybridTwoStageModel(nn.Module):
    """Two-model hybrid pipeline.

    Stage 1: predict future amplitude tokens from input amp+morph tokens.
    Stage 2: predict future morphology tokens from future amplitude tokens only.
    """

    def __init__(self, cfg: HybridTwoStageConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.amp_model = OperatorLearningModel(cfg.amp)
        self.morph_model = OperatorLearningModel(cfg.morph)
        self.morph_lead_time_mode = str(cfg.morph_lead_time_mode).lower()

        # Freeze unused heads/embeddings so DDP does not expect gradients for them.
        self._freeze_module(self.amp_model.morph_head)
        self._freeze_module(self.morph_model.amp_head)
        if str(cfg.morph.fusion).lower() == "amp_only":
            self._freeze_module(self.morph_model.embed_morph)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def _build_var_ids(slot_var_ids: torch.Tensor) -> torch.Tensor:
        # slot_var_ids are 1-based with 0 for inactive slots.
        return slot_var_ids.clamp(min=1).sub(1).clamp(min=0)

    def _resolve_morph_time_indices(
        self,
        input_time_idx: torch.Tensor,
        output_time_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.morph_lead_time_mode
        if mode == "zero":
            # Predict morphology at the same time as the (future) amplitude.
            return output_time_idx, output_time_idx
        if mode == "same":
            return input_time_idx, output_time_idx
        raise ValueError(f"Unsupported morph_lead_time_mode={mode}")

    def forward(
        self,
        source_amp: torch.Tensor,
        source_morph: torch.Tensor,
        input_time_idx: torch.Tensor,
        output_time_idx: torch.Tensor,
        active_var_mask: torch.Tensor,
        slot_var_ids: torch.Tensor,
        target_amp: torch.Tensor | None = None,
        morph_source: str = "pred",
        problem_type_id: torch.Tensor | None = None,
    ) -> HybridTwoStageOutput:
        if problem_type_id is None:
            problem_type_id = torch.zeros((source_amp.shape[0],), dtype=torch.long, device=source_amp.device)

        input_var_ids = self._build_var_ids(slot_var_ids)
        output_var_ids = input_var_ids

        amp_logits, _ = self.amp_model(
            input_amp=source_amp,
            input_morph=source_morph,
            input_time_idx=input_time_idx,
            output_time_idx=output_time_idx,
            input_var_ids=input_var_ids,
            output_var_ids=output_var_ids,
            input_var_mask=active_var_mask,
            output_var_mask=active_var_mask,
            problem_type_id=problem_type_id,
        )

        bsz, n_vars, h, w = source_amp.shape
        pred_amp = amp_logits.argmax(dim=-1).view(bsz, n_vars, h, w)

        morph_source = str(morph_source).lower()
        if morph_source == "gt":
            if target_amp is None:
                raise ValueError("morph_source='gt' requires target_amp")
            morph_input_amp = target_amp
        elif morph_source == "pred":
            morph_input_amp = pred_amp
        else:
            raise ValueError(f"Unsupported morph_source={morph_source}")

        morph_input_morph = torch.zeros_like(morph_input_amp)
        morph_in_time, morph_out_time = self._resolve_morph_time_indices(
            input_time_idx=input_time_idx,
            output_time_idx=output_time_idx,
        )

        _, morph_logits = self.morph_model(
            input_amp=morph_input_amp,
            input_morph=morph_input_morph,
            input_time_idx=morph_in_time,
            output_time_idx=morph_out_time,
            input_var_ids=input_var_ids,
            output_var_ids=output_var_ids,
            input_var_mask=active_var_mask,
            output_var_mask=active_var_mask,
            problem_type_id=problem_type_id,
        )
        pred_morph = morph_logits.argmax(dim=-1).view(bsz, n_vars, h, w)

        return HybridTwoStageOutput(
            logits_amp=amp_logits,
            logits_morph=morph_logits,
            pred_amp=pred_amp,
            pred_morph=pred_morph,
        )

    def predict_tokens(
        self,
        source_amp: torch.Tensor,
        source_morph: torch.Tensor,
        lead_time_idx: torch.Tensor,
        active_var_mask: torch.Tensor,
        slot_var_ids: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del spatial_mask
        input_time_idx = torch.zeros_like(lead_time_idx)
        output_time_idx = lead_time_idx

        out = self.forward(
            source_amp=source_amp,
            source_morph=source_morph,
            input_time_idx=input_time_idx,
            output_time_idx=output_time_idx,
            active_var_mask=active_var_mask,
            slot_var_ids=slot_var_ids,
            target_amp=None,
            morph_source="pred",
        )
        return out.pred_amp, out.pred_morph


def _build_operator_cfg(raw_cfg: dict[str, Any], ds_cfg: dict[str, Any]) -> OperatorConfig:
    legacy_decoder_depth = int(raw_cfg.get("decoder_depth", 3))
    return OperatorConfig(
        token_type="phaedra",
        vocab_amp=int(ds_cfg["amp_vocab_size"]),
        vocab_morph=int(ds_cfg["morph_vocab_size"]),
        vocab_fsq=None,
        num_in_vars=len(ds_cfg["input_variables"]),
        num_out_vars=len(ds_cfg["output_variables"]),
        grid_size=int(raw_cfg.get("grid_size", 32)),
        embed_dim=int(raw_cfg.get("embed_dim", 256)),
        encoder_depth=int(raw_cfg.get("encoder_depth", 18)),
        amp_decoder_depth=int(raw_cfg.get("amp_decoder_depth", legacy_decoder_depth)),
        morph_decoder_depth=int(raw_cfg.get("morph_decoder_depth", legacy_decoder_depth)),
        num_heads=int(raw_cfg.get("num_heads", 8)),
        mlp_ratio=int(raw_cfg.get("mlp_ratio", 4)),
        fusion=str(raw_cfg.get("fusion", "concat")),
        layerscale_init=float(raw_cfg.get("layerscale_init", 1e-5)),
        max_time_index=int(raw_cfg.get("max_time_index", 14)),
        problem_type_embed_scale=float(raw_cfg.get("problem_type_embed_scale", 1.0)),
    )


def build_hybrid_two_stage_config(raw_cfg: dict[str, Any], ds_cfg: dict[str, Any]) -> HybridTwoStageConfig:
    if "amp" not in raw_cfg or "morph" not in raw_cfg:
        raise ValueError("Hybrid model config must include model.amp and model.morph sections")
    amp_cfg = _build_operator_cfg(raw_cfg["amp"], ds_cfg)
    morph_cfg = _build_operator_cfg(raw_cfg["morph"], ds_cfg)
    return HybridTwoStageConfig(
        amp=amp_cfg,
        morph=morph_cfg,
        morph_lead_time_mode=str(raw_cfg.get("morph_lead_time_mode", "zero")),
    )
