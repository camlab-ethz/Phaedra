"""Two-stage seq2seq operator: amp -> morph(amp).

Subclasses `hub.models.seq2seq_model.OperatorLearningModel` and overrides the
decoder pass so that morph is conditioned on the predicted amplitude (either GT
in mode A or soft / probability-weighted in mode B). The encoder, embedding
tables, attention blocks and head heads are reused unchanged.

`amp_temperature` controls the softness used in mode B; default 1.0.

Subclasses change which encoder/decoder blocks are used (linear / longformer)
by overriding `_make_encoder` / `_make_decoder`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hub.models.seq2seq_model import (
    OperatorConfig,
    OperatorLearningModel,
)

from .attention_blocks import FlashDecoderBlock, FlashEncoderBlock


@dataclass
class TwoStageOutput:
    amp_logits: torch.Tensor    # [B, V*H*W, vocab_amp]
    morph_logits: torch.Tensor  # [B, V*H*W, vocab_morph]
    pred_amp: torch.Tensor      # [B, V*H*W]
    pred_morph: torch.Tensor    # [B, V*H*W]


class TwoStageSeq2SeqOperator(OperatorLearningModel):
    """Run the decoder twice. Stage 1 predicts amp; stage 2 predicts morph with
    the stage-1 amplitude added back to the decoder input as conditioning.

    Mode A: stage 2 uses GT amp embeddings (teacher forced) -- the typical
            training path.
    Mode B: stage 2 uses softmax-weighted amp embeddings from stage 1 logits
            (`amp_temperature` controls sharpness) -- used at validation /
            inference so no GT amp is required.
    """

    def __init__(self, cfg: OperatorConfig, amp_temperature: float = 1.0):
        super().__init__(cfg)
        self.amp_temperature = float(amp_temperature)
        # Stage tag to differentiate the two decoder passes. Adds 2*d params.
        self.stage_embed = nn.Embedding(2, cfg.embed_dim)
        nn.init.trunc_normal_(self.stage_embed.weight, std=0.02)

        # Replace the parent's masked-SDPA encoder/decoder with the mask-free
        # Flash blocks (parameter-identical drop-ins that pass `attn_mask=None`
        # to SDPA, avoiding the torch.compile -> math backend -> fp32 QK^T OOM).
        # Variant subclasses can override either hook to plug in linear /
        # Longformer / etc. attention blocks instead.
        self.encoder = nn.ModuleList(self._make_encoder(cfg))
        self.decoder = nn.ModuleList(self._make_decoder(cfg))

    def _make_encoder(self, cfg: OperatorConfig):
        return [
            FlashEncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
            for _ in range(cfg.encoder_depth)
        ]

    def _make_decoder(self, cfg: OperatorConfig):
        return [
            FlashDecoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
            for _ in range(cfg.decoder_depth)
        ]

    # ------------------------------------------------------------------
    # Internals lifted from OperatorLearningModel.forward, broken into
    # encode + decode so we can run the decoder twice.
    # ------------------------------------------------------------------
    def _encode(
        self,
        input_amp: torch.Tensor,
        input_morph: torch.Tensor,
        input_time_idx: torch.Tensor,
        output_time_idx: torch.Tensor,
        input_var_ids: torch.Tensor | None,
        input_var_mask: torch.Tensor | None,
        problem_type_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_vars, h, w = input_amp.shape
        if h != self.grid_size or w != self.grid_size:
            raise ValueError(f"Expected grid_size={self.grid_size}, got {(h, w)}")

        delta_t = (output_time_idx.long() - input_time_idx.long()).clamp(min=0, max=self.cfg.max_time_index)
        if problem_type_id is None:
            problem_type_id = torch.zeros((bsz,), dtype=torch.long, device=input_amp.device)
        problem_type_id = problem_type_id.clamp(min=0, max=self.problem_type_embed.num_embeddings - 1)
        cond_emb = self.lead_time_embed(delta_t) + self.problem_type_embed(problem_type_id) * self.problem_type_embed_scale

        in_amp_flat = input_amp.reshape(bsz, num_vars, h * w)
        in_morph_flat = input_morph.reshape(bsz, num_vars, h * w)
        embeds = self._embed_tokens(in_amp_flat, in_morph_flat).view(bsz, num_vars * h * w, -1)

        if input_var_ids is None:
            input_var_ids = torch.arange(num_vars, dtype=torch.long, device=embeds.device).unsqueeze(0).expand(bsz, -1)
        input_var_ids = input_var_ids.clamp(min=0, max=self.input_var_embed.num_embeddings - 1)
        in_var_flat = input_var_ids[:, :, None].expand(bsz, num_vars, h * w).reshape(bsz, num_vars * h * w)
        embeds = embeds + self.input_var_embed(in_var_flat)

        if input_var_mask is None:
            input_var_mask = torch.ones((bsz, num_vars), dtype=torch.bool, device=embeds.device)
        enc_mask = input_var_mask[:, :, None].expand(bsz, num_vars, h * w).reshape(bsz, num_vars * h * w)

        spatial = torch.arange(h * w, device=embeds.device, dtype=torch.long).repeat(num_vars).unsqueeze(0).expand(bsz, -1)

        x = embeds + cond_emb.unsqueeze(1)
        for blk in self.encoder:
            x = blk(x, attention_mask=enc_mask, spatial_indices=spatial)
        return x, enc_mask, cond_emb, spatial

    def _decode(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cond_emb: torch.Tensor,
        bsz: int,
        h: int,
        w: int,
        num_out_vars: int,
        output_var_ids: torch.Tensor | None,
        output_var_mask: torch.Tensor | None,
        stage_id: int,
        amp_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        out_len = num_out_vars * h * w
        device = memory.device

        if output_var_mask is None:
            output_var_mask = torch.ones((bsz, num_out_vars), dtype=torch.bool, device=device)
        dec_mask = output_var_mask[:, :, None].expand(bsz, num_out_vars, h * w).reshape(bsz, out_len)

        if output_var_ids is None:
            output_var_ids = torch.arange(num_out_vars, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
        output_var_ids = output_var_ids.clamp(min=0, max=self.output_var_embed.num_embeddings - 1)
        out_var_flat = output_var_ids[:, :, None].expand(bsz, num_out_vars, h * w).reshape(bsz, out_len)

        stage_vec = self.stage_embed.weight[stage_id].view(1, 1, -1)
        dec_x = self.output_token.repeat(bsz, out_len, 1)
        dec_x = dec_x + self.output_var_embed(out_var_flat) + cond_emb.unsqueeze(1) + stage_vec
        if amp_condition is not None:
            dec_x = dec_x + amp_condition

        out_spatial = torch.arange(h * w, device=device, dtype=torch.long).repeat(num_out_vars).unsqueeze(0).expand(bsz, -1)

        for blk in self.decoder:
            dec_x = blk(
                dec_x, memory=memory, tgt_mask=dec_mask,
                memory_mask=memory_mask, tgt_spatial_indices=out_spatial,
            )
        return dec_x

    def forward(
        self,
        input_amp: torch.Tensor | None = None,
        input_morph: torch.Tensor | None = None,
        input_tokens: torch.Tensor | None = None,
        input_time_idx: torch.Tensor | None = None,
        output_time_idx: torch.Tensor | None = None,
        input_var_ids: torch.Tensor | None = None,
        output_var_ids: torch.Tensor | None = None,
        input_var_mask: torch.Tensor | None = None,
        output_var_mask: torch.Tensor | None = None,
        problem_type_id: torch.Tensor | None = None,
        target_amp: torch.Tensor | None = None,
        mode: str = "A",
    ) -> TwoStageOutput:
        del input_tokens
        if input_amp is None or input_morph is None:
            raise ValueError("forward requires input_amp and input_morph")
        if input_time_idx is None or output_time_idx is None:
            raise ValueError("forward requires input_time_idx and output_time_idx")

        mode = mode.upper()
        if mode not in {"A", "B"}:
            raise ValueError(f"Unsupported mode={mode}; expected 'A' or 'B'.")
        if mode == "A" and target_amp is None:
            raise ValueError("Mode A requires target_amp for teacher-forced amp conditioning.")

        bsz, _, h, w = input_amp.shape
        memory, memory_mask, cond_emb, _ = self._encode(
            input_amp, input_morph, input_time_idx, output_time_idx,
            input_var_ids, input_var_mask, problem_type_id,
        )

        if output_var_ids is not None:
            num_out_vars = int(output_var_ids.shape[1])
        elif output_var_mask is not None:
            num_out_vars = int(output_var_mask.shape[1])
        else:
            num_out_vars = self.num_out_vars

        # Stage 1: predict amp.
        ctx_amp = self._decode(
            memory, memory_mask, cond_emb, bsz, h, w, num_out_vars,
            output_var_ids, output_var_mask, stage_id=0, amp_condition=None,
        )
        amp_logits = self.head_amp(ctx_amp)
        pred_amp = amp_logits.argmax(dim=-1)

        # Stage 2: morph, conditioned on (mode-A: GT, mode-B: soft) amp.
        if mode == "A":
            tgt_flat = target_amp.permute(0, 2, 3, 1).reshape(bsz, num_out_vars * h * w)
            amp_cond = self.embed_amp(tgt_flat)
        else:
            probs = F.softmax(amp_logits / max(self.amp_temperature, 1e-6), dim=-1)
            amp_cond = probs @ self.embed_amp.weight

        ctx_morph = self._decode(
            memory, memory_mask, cond_emb, bsz, h, w, num_out_vars,
            output_var_ids, output_var_mask, stage_id=1, amp_condition=amp_cond,
        )
        morph_logits = self.head_morph(ctx_morph)
        pred_morph = morph_logits.argmax(dim=-1)

        return TwoStageOutput(
            amp_logits=amp_logits,
            morph_logits=morph_logits,
            pred_amp=pred_amp,
            pred_morph=pred_morph,
        )


def build_operator_config(cfg: dict[str, Any], fusion: str) -> OperatorConfig:
    """Compose an OperatorConfig from a benchmark YAML, fixing the fusion mode."""
    ds_cfg = cfg["dataset"]
    raw = dict(cfg["model"])
    return OperatorConfig(
        token_type="phaedra",
        vocab_amp=int(ds_cfg["amp_vocab_size"]),
        vocab_morph=int(ds_cfg["morph_vocab_size"]),
        vocab_fsq=None,
        num_in_vars=int(len(ds_cfg["input_variables"])),
        num_out_vars=int(len(ds_cfg["output_variables"])),
        grid_size=int(raw["grid_size"]),
        embed_dim=int(raw["embed_dim"]),
        encoder_depth=int(raw["encoder_depth"]),
        decoder_depth=int(raw["decoder_depth"]),
        num_heads=int(raw["num_heads"]),
        mlp_ratio=int(raw.get("mlp_ratio", 4)),
        fusion=fusion,
        layerscale_init=float(raw.get("layerscale_init", 1e-5)),
        max_time_index=int(raw.get("max_time_index", 14)),
        problem_type_embed_scale=float(raw.get("problem_type_embed_scale", 1.0)),
    )
