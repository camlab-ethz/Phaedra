"""Dual-decoder seq2seq operator: separate amp and morph decoder stacks.

Two architectural variants share this code:

  - **parallel**: `amp_decoder` and `morph_decoder` each cross-attend to encoder
    memory independently. No information flows from amp predictions to morph.
  - **sequential**: `amp_decoder` predicts amp; the soft amp embedding
    `softmax(amp_logits / T) @ embed_amp.weight` is added to the
    `morph_decoder`'s input as per-position conditioning, so the morph head
    sees the predicted amp at every spatial slot.

For the sequential variant we always use the soft-amp signal (no scheduled
teacher forcing). The soft signal is a natural curriculum: early in training
amp logits are noisy -> soft amp is close to the mean amp embedding (weak
signal, morph learns from the encoder); as amp head improves the soft signal
sharpens into a near-one-hot and morph gets a strong constraint. No train/test
discrepancy, no extra hyperparameter.

Source-side architecture matches our existing `flash_concat` variant exactly:
FlashSDPA attention, `Linear([amp_e; morph_e], d)` per-(var, position) fusion,
source sequence length V*H*W = 4096. The shared encoder is reused by both
decoders.

A typical config has the encoder ~3-4x deeper than each decoder stack, e.g.:

    embed_dim=288, encoder_depth=19, amp_decoder_depth=5, morph_decoder_depth=5

which lands at ~38M parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_blocks import FlashDecoderBlock, FlashEncoderBlock
from .two_stage_seq2seq import TwoStageOutput


@dataclass
class DualDecoderConfig:
    vocab_amp: int
    vocab_morph: int
    num_in_vars: int
    num_out_vars: int
    grid_size: int
    embed_dim: int
    encoder_depth: int
    amp_decoder_depth: int
    morph_decoder_depth: int
    num_heads: int
    mlp_ratio: int = 4
    fusion: str = "concat"        # 'concat' (Linear([amp; morph], d)) or 'sum'
    max_time_index: int = 14
    problem_type_embed_scale: float = 1.0
    amp_temperature: float = 1.0
    use_amp_conditioning: bool = False   # True => sequential, False => parallel
    # If True, add a learned [H*W, embed_dim] positional embedding to the
    # decoder query (shared between amp and morph stacks, indexed by spatial
    # position). Default False preserves the original architecture so existing
    # checkpoints stay byte-compatible and apples-to-apples comparisons hold.
    # Set True for synthetic procedural pretraining where the decoder query
    # otherwise starts position-blind and cross-attention can't escape the
    # uniform-attention saddle on random inputs -- see synthetic.py docstring.
    use_decoder_pos_embed: bool = False


class DualDecoderSeq2SeqOperator(nn.Module):
    """Shared encoder + two independent decoder stacks (amp, morph) sharing the
    same vocab embeddings and conditioning embeddings.

    Always uses the mask-free `FlashEncoderBlock` / `FlashDecoderBlock` so SDPA
    stays on the Flash backend under `torch.compile` (no fp32 score-matrix
    materialisation -- see attention_blocks.py for the rationale).
    """

    def __init__(self, cfg: DualDecoderConfig):
        super().__init__()
        self.cfg = cfg
        self.use_amp_conditioning = bool(cfg.use_amp_conditioning)
        self.grid_size = int(cfg.grid_size)
        self.num_in_vars = int(cfg.num_in_vars)
        self.num_out_vars = int(cfg.num_out_vars)
        self.problem_type_embed_scale = float(cfg.problem_type_embed_scale)
        d = int(cfg.embed_dim)

        # Vocab + source fusion.
        self.embed_amp = nn.Embedding(cfg.vocab_amp, d)
        self.embed_morph = nn.Embedding(cfg.vocab_morph, d)
        if cfg.fusion == "concat":
            self.fusion_proj = nn.Linear(2 * d, d)
        elif cfg.fusion == "sum":
            self.fusion_proj = None
        else:
            raise ValueError(f"Unsupported fusion={cfg.fusion!r}")

        # Conditioning embeddings (same names/shapes as OperatorLearningModel
        # for ease of comparison with the other variants).
        self.lead_time_embed = nn.Embedding(cfg.max_time_index + 1, d)
        self.input_var_embed = nn.Embedding(cfg.num_in_vars + 1, d)
        self.output_var_embed = nn.Embedding(cfg.num_out_vars + 1, d)
        self.problem_type_embed = nn.Embedding(4, d)

        # Shared encoder.
        self.encoder = nn.ModuleList([
            FlashEncoderBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
            for _ in range(cfg.encoder_depth)
        ])

        # Two separate decoder stacks. Same block class; independent weights.
        self.amp_decoder = nn.ModuleList([
            FlashDecoderBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
            for _ in range(cfg.amp_decoder_depth)
        ])
        self.morph_decoder = nn.ModuleList([
            FlashDecoderBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
            for _ in range(cfg.morph_decoder_depth)
        ])

        # Separate decoder start tokens.
        self.amp_start = nn.Parameter(torch.zeros(1, 1, d))
        self.morph_start = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.amp_start, std=0.02)
        nn.init.trunc_normal_(self.morph_start, std=0.02)

        # Optional learned 2D positional embedding for the decoder query.
        # Shared between amp and morph stacks (same "this is position (h, w)"
        # meaning). Without this, the decoder query starts position-blind
        # (`start_token + output_var_embed + cond_emb` is the same at every
        # spatial slot within a variable) and cross-attention cannot break
        # symmetry on random synthetic data.
        self.use_decoder_pos_embed = bool(cfg.use_decoder_pos_embed)
        if self.use_decoder_pos_embed:
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(self.grid_size * self.grid_size, d)
            )
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        else:
            self.decoder_pos_embed = None

        # Output heads (tied? -> kept untied to match OperatorLearningModel).
        self.head_amp = nn.Linear(d, cfg.vocab_amp)
        self.head_morph = nn.Linear(d, cfg.vocab_morph)

    # ------------------------------------------------------------------
    # Source fusion (per-(var, position) concat-and-project or sum).
    # ------------------------------------------------------------------
    def _embed_source_tokens(self, amp_tokens: torch.Tensor, morph_tokens: torch.Tensor) -> torch.Tensor:
        amp_e = self.embed_amp(amp_tokens)
        morph_e = self.embed_morph(morph_tokens)
        if self.fusion_proj is not None:
            return self.fusion_proj(torch.cat([amp_e, morph_e], dim=-1))
        return amp_e + morph_e

    # ------------------------------------------------------------------
    # Shared encoder forward.
    # ------------------------------------------------------------------
    def _encode(
        self,
        input_amp: torch.Tensor,
        input_morph: torch.Tensor,
        input_time_idx: torch.Tensor,
        output_time_idx: torch.Tensor,
        input_var_ids: torch.Tensor | None,
        problem_type_id: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_vars, h, w = input_amp.shape
        if h != self.grid_size or w != self.grid_size:
            raise ValueError(f"Expected grid_size={self.grid_size}, got {(h, w)}")
        device = input_amp.device

        delta_t = (output_time_idx.long() - input_time_idx.long()).clamp(
            min=0, max=self.cfg.max_time_index
        )
        if problem_type_id is None:
            problem_type_id = torch.zeros((bsz,), dtype=torch.long, device=device)
        problem_type_id = problem_type_id.clamp(min=0, max=self.problem_type_embed.num_embeddings - 1)
        cond_emb = (
            self.lead_time_embed(delta_t)
            + self.problem_type_embed(problem_type_id) * self.problem_type_embed_scale
        )

        in_amp_flat = input_amp.reshape(bsz, num_vars, h * w)
        in_morph_flat = input_morph.reshape(bsz, num_vars, h * w)
        embeds = self._embed_source_tokens(in_amp_flat, in_morph_flat).view(bsz, num_vars * h * w, -1)

        if input_var_ids is None:
            input_var_ids = (
                torch.arange(num_vars, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        input_var_ids = input_var_ids.clamp(min=0, max=self.input_var_embed.num_embeddings - 1)
        in_var_flat = (
            input_var_ids[:, :, None]
            .expand(bsz, num_vars, h * w)
            .reshape(bsz, num_vars * h * w)
        )
        embeds = embeds + self.input_var_embed(in_var_flat)

        enc_mask = torch.ones((bsz, num_vars * h * w), dtype=torch.bool, device=device)
        spatial = (
            torch.arange(h * w, device=device, dtype=torch.long)
            .repeat(num_vars)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        x = embeds + cond_emb.unsqueeze(1)
        for blk in self.encoder:
            x = blk(x, attention_mask=enc_mask, spatial_indices=spatial)
        return x, enc_mask, cond_emb

    # ------------------------------------------------------------------
    # Single decoder-stack forward (used for both amp and morph stages).
    # ------------------------------------------------------------------
    def _run_decoder(
        self,
        decoder: nn.ModuleList,
        start_token: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        cond_emb: torch.Tensor,
        bsz: int,
        h: int,
        w: int,
        num_out_vars: int,
        output_var_ids: torch.Tensor | None,
        output_var_mask: torch.Tensor | None,
        amp_condition: torch.Tensor | None,
    ) -> torch.Tensor:
        out_len = num_out_vars * h * w
        device = memory.device

        if output_var_mask is None:
            output_var_mask = torch.ones((bsz, num_out_vars), dtype=torch.bool, device=device)
        dec_mask = (
            output_var_mask[:, :, None]
            .expand(bsz, num_out_vars, h * w)
            .reshape(bsz, out_len)
        )

        if output_var_ids is None:
            output_var_ids = (
                torch.arange(num_out_vars, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        output_var_ids = output_var_ids.clamp(min=0, max=self.output_var_embed.num_embeddings - 1)
        out_var_flat = (
            output_var_ids[:, :, None]
            .expand(bsz, num_out_vars, h * w)
            .reshape(bsz, out_len)
        )

        dec_x = start_token.repeat(bsz, out_len, 1)
        dec_x = dec_x + self.output_var_embed(out_var_flat) + cond_emb.unsqueeze(1)
        if amp_condition is not None:
            dec_x = dec_x + amp_condition

        out_spatial = (
            torch.arange(h * w, device=device, dtype=torch.long)
            .repeat(num_out_vars)
            .unsqueeze(0)
            .expand(bsz, -1)
        )

        # Add the optional 2D positional embedding so the cross-attention
        # query is position-aware from step 0 rather than depending on
        # decoder self-attention + RoPE to develop it (the source of the
        # synthetic-data cold-start failure). Broadcasts cleanly:
        # decoder_pos_embed[out_spatial] -> [bsz, out_len, d].
        if self.decoder_pos_embed is not None:
            dec_x = dec_x + self.decoder_pos_embed[out_spatial]
        for blk in decoder:
            dec_x = blk(
                dec_x, memory=memory, tgt_mask=dec_mask,
                memory_mask=memory_mask, tgt_spatial_indices=out_spatial,
            )
        return dec_x

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
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
        mode: str = "B",
    ) -> TwoStageOutput:
        del input_tokens, input_var_mask  # all True for our benchmark datasets

        if input_amp is None or input_morph is None:
            raise ValueError("forward requires input_amp and input_morph")
        if input_time_idx is None or output_time_idx is None:
            raise ValueError("forward requires input_time_idx and output_time_idx")

        bsz, _, h, w = input_amp.shape
        memory, memory_mask, cond_emb = self._encode(
            input_amp, input_morph, input_time_idx, output_time_idx,
            input_var_ids, problem_type_id,
        )

        if output_var_ids is not None:
            num_out_vars = int(output_var_ids.shape[1])
        elif output_var_mask is not None:
            num_out_vars = int(output_var_mask.shape[1])
        else:
            num_out_vars = self.num_out_vars

        # Stage 1: amp decoder.
        ctx_amp = self._run_decoder(
            self.amp_decoder, self.amp_start, memory, memory_mask, cond_emb,
            bsz, h, w, num_out_vars, output_var_ids, output_var_mask,
            amp_condition=None,
        )
        amp_logits = self.head_amp(ctx_amp)
        pred_amp = amp_logits.argmax(dim=-1)

        # Stage 2: morph decoder. Conditioning is only added in the sequential
        # variant. We pre-compute the amp embedding the morph decoder will see
        # and pipe it through the same code path as the parallel variant.
        amp_condition: torch.Tensor | None = None
        if self.use_amp_conditioning:
            mode_upper = mode.upper()
            if mode_upper == "A":
                if target_amp is None:
                    raise ValueError("Mode A (teacher-forced amp) requires target_amp.")
                tgt_flat = target_amp.permute(0, 2, 3, 1).reshape(bsz, num_out_vars * h * w)
                amp_condition = self.embed_amp(tgt_flat)
            else:
                probs = F.softmax(amp_logits / max(self.cfg.amp_temperature, 1e-6), dim=-1)
                amp_condition = probs @ self.embed_amp.weight

        ctx_morph = self._run_decoder(
            self.morph_decoder, self.morph_start, memory, memory_mask, cond_emb,
            bsz, h, w, num_out_vars, output_var_ids, output_var_mask,
            amp_condition=amp_condition,
        )
        morph_logits = self.head_morph(ctx_morph)
        pred_morph = morph_logits.argmax(dim=-1)

        return TwoStageOutput(
            amp_logits=amp_logits,
            morph_logits=morph_logits,
            pred_amp=pred_amp,
            pred_morph=pred_morph,
        )

    def predict(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper compatible with `hub.trainers.test._predict_step_seq2seq`.

        Always runs in mode "B" (soft amp conditioning): no GT amp tokens are
        needed, the morph decoder consumes the softmax-weighted amp embedding
        from the amp head's own logits. For the `dual_parallel` variant the
        conditioning is ignored entirely, so the mode is a no-op there.
        """
        del input_tokens
        with torch.no_grad():
            out = self.forward(
                input_amp=input_amp,
                input_morph=input_morph,
                input_time_idx=input_time_idx,
                output_time_idx=output_time_idx,
                input_var_ids=input_var_ids,
                output_var_ids=output_var_ids,
                input_var_mask=input_var_mask,
                output_var_mask=output_var_mask,
                problem_type_id=problem_type_id,
                target_amp=None,
                mode="B",
            )
        return out.pred_amp, out.pred_morph


def _config_from_dict(cfg: dict[str, Any], use_amp_conditioning: bool) -> DualDecoderConfig:
    ds = cfg["dataset"]
    raw = dict(cfg["model"])
    return DualDecoderConfig(
        vocab_amp=int(ds["amp_vocab_size"]),
        vocab_morph=int(ds["morph_vocab_size"]),
        num_in_vars=int(len(ds["input_variables"])),
        num_out_vars=int(len(ds["output_variables"])),
        grid_size=int(raw["grid_size"]),
        embed_dim=int(raw["embed_dim"]),
        encoder_depth=int(raw["encoder_depth"]),
        amp_decoder_depth=int(raw["amp_decoder_depth"]),
        morph_decoder_depth=int(raw["morph_decoder_depth"]),
        num_heads=int(raw["num_heads"]),
        mlp_ratio=int(raw.get("mlp_ratio", 4)),
        fusion=str(raw.get("fusion", "concat")),
        max_time_index=int(raw.get("max_time_index", 14)),
        problem_type_embed_scale=float(raw.get("problem_type_embed_scale", 1.0)),
        amp_temperature=float(raw.get("amp_temperature", 1.0)),
        use_amp_conditioning=use_amp_conditioning,
        use_decoder_pos_embed=bool(raw.get("use_decoder_pos_embed", False)),
    )


def build_dual_parallel(cfg: dict[str, Any]) -> nn.Module:
    """Two independent decoders. Morph cross-attends to encoder memory only;
    no amp -> morph information flow."""
    return DualDecoderSeq2SeqOperator(_config_from_dict(cfg, use_amp_conditioning=False))


def build_dual_sequential(cfg: dict[str, Any]) -> nn.Module:
    """Two decoders + soft amp conditioning. Amp decoder runs first; the soft
    amp embedding (softmax(amp_logits / T) @ embed_amp.weight) is added to the
    morph decoder's input as per-position conditioning."""
    return DualDecoderSeq2SeqOperator(_config_from_dict(cfg, use_amp_conditioning=True))
