from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope_utils import apply_2d_rope
from .attention2d import make_self_attention2d


@dataclass
class OperatorConfig:
    token_type: str
    vocab_amp: int | None
    vocab_morph: int | None
    vocab_fsq: int | None
    num_in_vars: int
    num_out_vars: int
    grid_size: int
    embed_dim: int
    encoder_depth: int
    amp_decoder_depth: int
    morph_decoder_depth: int
    num_heads: int
    mlp_ratio: int
    fusion: str
    layerscale_init: float | None
    max_time_index: int
    problem_type_embed_scale: float = 1.0
    attention_mode: str = "full"  # encoder self-attention: full(oracle)|linear|windowed|ssm|radial|quadtree


class RoPEEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, grid_size: int, attention_mode: str = "full") -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_width = grid_size
        self.attention_mode = str(attention_mode).lower().strip()

        self.norm1 = nn.LayerNorm(embed_dim)
        if self.attention_mode == "full":
            # oracle: inline full SDPA. Param names unchanged so existing
            # seq2seq checkpoints keep loading.
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.attn = make_self_attention2d(self.attention_mode, embed_dim, num_heads, grid_size)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        if self.attention_mode == "full":
            q = self._shape(self.q_proj(h))
            k = self._shape(self.k_proj(h))
            v = self._shape(self.v_proj(h))
            q, k = apply_2d_rope(q, k, spatial_indices=spatial_indices, grid_width=self.grid_width)
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask[:, None, None, :].to(torch.bool),
                dropout_p=0.0,
            )
            attn_out = attn_out.transpose(1, 2).reshape(x.shape)
            x = x + self.out_proj(attn_out)
        else:
            x = x + self.attn(h, attention_mask, spatial_indices)

        x = x + self.mlp(self.norm2(x))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


class RoPEDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, grid_size: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_width = grid_size

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_q = nn.Linear(embed_dim, embed_dim)
        self.self_k = nn.Linear(embed_dim, embed_dim)
        self.self_v = nn.Linear(embed_dim, embed_dim)
        self.self_out = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_q = nn.Linear(embed_dim, embed_dim)
        self.cross_k = nn.Linear(embed_dim, embed_dim)
        self.cross_v = nn.Linear(embed_dim, embed_dim)
        self.cross_out = nn.Linear(embed_dim, embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt_spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        q = self._shape(self.self_q(h))
        k = self._shape(self.self_k(h))
        v = self._shape(self.self_v(h))
        q, k = apply_2d_rope(q, k, spatial_indices=tgt_spatial_indices, grid_width=self.grid_width)

        self_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=tgt_mask[:, None, None, :].to(torch.bool),
            dropout_p=0.0,
        )
        self_out = self_out.transpose(1, 2).reshape(x.shape)
        x = x + self.self_out(self_out)

        h = self.norm2(x)
        q = self._shape(self.cross_q(h))
        k = self._shape(self.cross_k(memory))
        v = self._shape(self.cross_v(memory))
        cross_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=memory_mask[:, None, None, :].to(torch.bool),
            dropout_p=0.0,
        )
        cross_out = cross_out.transpose(1, 2).reshape(x.shape)
        x = x + self.cross_out(cross_out)

        x = x + self.mlp(self.norm3(x))
        return x * tgt_mask.unsqueeze(-1).to(x.dtype)


class OperatorLearningModel(nn.Module):
    def __init__(self, cfg: OperatorConfig):
        super().__init__()
        self.cfg = cfg
        self.token_type = str(cfg.token_type).lower().strip()
        self.grid_size = cfg.grid_size
        self.num_in_vars = cfg.num_in_vars
        self.num_out_vars = cfg.num_out_vars
        self.fusion = cfg.fusion
        self.problem_type_embed_scale = float(cfg.problem_type_embed_scale)

        if self.token_type == "phaedra":
            if cfg.vocab_amp is None or cfg.vocab_morph is None:
                raise ValueError("Phaedra vocab sizes required")

            self.embed_amp = nn.Embedding(cfg.vocab_amp, cfg.embed_dim)
            self.embed_morph = nn.Embedding(cfg.vocab_morph, cfg.embed_dim)
            self.fusion_proj = nn.Linear(cfg.embed_dim * 2, cfg.embed_dim) if cfg.fusion == "concat" else None
        elif self.token_type == "fsq":
            if cfg.vocab_fsq is None:
                raise ValueError("FSQ vocab size required")
            self.embed_fsq = nn.Embedding(cfg.vocab_fsq, cfg.embed_dim)
            self.fusion_proj = None
        else:
            raise ValueError(f"Unsupported token_type for seq2seq: {self.token_type}")

        # Conditioning terms are added to token embeddings (original behavior).
        self.lead_time_embed = nn.Embedding(cfg.max_time_index + 1, cfg.embed_dim)
        self.input_var_embed = nn.Embedding(cfg.num_in_vars + 1, cfg.embed_dim)
        self.output_var_embed = nn.Embedding(cfg.num_out_vars + 1, cfg.embed_dim)
        self.problem_type_embed = nn.Embedding(4, cfg.embed_dim)

        self.encoder = nn.ModuleList(
            [RoPEEncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size, attention_mode=cfg.attention_mode) for _ in range(cfg.encoder_depth)]
        )
        if self.token_type == "phaedra":
            self.amplitude_decoder = nn.ModuleList(
                [
                    RoPEDecoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
                    for _ in range(cfg.amp_decoder_depth)
                ]
            )
            self.morphology_decoder = nn.ModuleList(
                [
                    RoPEDecoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
                    for _ in range(cfg.morph_decoder_depth)
                ]
            )
        else:
            self.decoder = nn.ModuleList(
                [
                    RoPEDecoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
                    for _ in range(cfg.amp_decoder_depth)
                ]
            )

        self.output_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.output_token, std=0.02)

        if self.token_type == "phaedra":
            self.amp_head = nn.Linear(cfg.embed_dim, cfg.vocab_amp)
            self.morph_head = nn.Linear(cfg.embed_dim, cfg.vocab_morph)

            # Backward-compatible aliases for older code paths.
            self.head_amp = self.amp_head
            self.head_morph = self.morph_head
        else:
            self.fsq_head = nn.Linear(cfg.embed_dim, cfg.vocab_fsq)
            self.head_fsq = self.fsq_head

    def _embed_tokens(self, tokens_amp: torch.Tensor, tokens_morph: torch.Tensor) -> torch.Tensor:
        if self.token_type != "phaedra":
            raise ValueError("_embed_tokens is only valid for phaedra token_type")
        amp_emb = self.embed_amp(tokens_amp)
        morph_emb = self.embed_morph(tokens_morph)
        if self.fusion == "concat":
            return self.fusion_proj(torch.cat([amp_emb, morph_emb], dim=-1))
        if self.fusion == "amp_only":
            return amp_emb
        if self.fusion == "morph_only":
            return morph_emb
        return amp_emb + morph_emb

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
    ):
        if self.token_type == "phaedra":
            if input_amp is None or input_morph is None:
                raise ValueError("forward requires input_amp and input_morph")
        else:
            if input_tokens is None:
                raise ValueError("forward requires input_tokens for fsq")
        if input_time_idx is None or output_time_idx is None:
            raise ValueError("forward requires input_time_idx and output_time_idx")

        if self.token_type == "phaedra":
            batch_size, num_vars, h, w = input_amp.shape
        else:
            batch_size, num_vars, h, w = input_tokens.shape
        if h != self.grid_size or w != self.grid_size:
            raise ValueError(f"Expected grid_size={self.grid_size}, got {(h, w)}")

        delta_t = (output_time_idx.long() - input_time_idx.long()).clamp(min=0, max=self.cfg.max_time_index)
        if problem_type_id is None:
            problem_type_id = torch.zeros((batch_size,), dtype=torch.long, device=input_amp.device)
        problem_type_id = problem_type_id.clamp(min=0, max=self.problem_type_embed.num_embeddings - 1)
        cond_emb = self.lead_time_embed(delta_t) + self.problem_type_embed(problem_type_id) * self.problem_type_embed_scale

        if self.token_type == "phaedra":
            input_amp = input_amp.reshape(batch_size, num_vars, h * w)
            input_morph = input_morph.reshape(batch_size, num_vars, h * w)
            embeds = self._embed_tokens(input_amp, input_morph).view(batch_size, num_vars * h * w, -1)
        else:
            input_tokens = input_tokens.reshape(batch_size, num_vars, h * w)
            embeds = self.embed_fsq(input_tokens).view(batch_size, num_vars * h * w, -1)

        if input_var_ids is None:
            input_var_ids = torch.arange(num_vars, dtype=torch.long, device=embeds.device).unsqueeze(0).expand(batch_size, -1)
        input_var_ids = input_var_ids.clamp(min=0, max=self.input_var_embed.num_embeddings - 1)
        input_var_flat = input_var_ids[:, :, None].expand(batch_size, num_vars, h * w).reshape(batch_size, num_vars * h * w)
        embeds = embeds + self.input_var_embed(input_var_flat)

        if input_var_mask is None:
            input_var_mask = torch.ones((batch_size, num_vars), dtype=torch.bool, device=embeds.device)
        enc_valid = input_var_mask[:, :, None].expand(batch_size, num_vars, h * w).reshape(batch_size, num_vars * h * w)

        in_spatial = torch.arange(h * w, device=embeds.device, dtype=torch.long).repeat(num_vars)
        in_spatial = in_spatial.unsqueeze(0).expand(batch_size, -1)

        enc_x = embeds + cond_emb.unsqueeze(1)
        enc_mask = enc_valid
        enc_spatial = in_spatial

        for blk in self.encoder:
            enc_x = blk(enc_x, attention_mask=enc_mask, spatial_indices=enc_spatial)
        memory = enc_x

        if output_var_ids is not None:
            num_out_vars = int(output_var_ids.shape[1])
        elif output_var_mask is not None:
            num_out_vars = int(output_var_mask.shape[1])
        else:
            num_out_vars = self.num_out_vars
        out_len = num_out_vars * h * w

        if output_var_mask is None:
            output_var_mask = torch.ones((batch_size, num_out_vars), dtype=torch.bool, device=embeds.device)
        dec_valid = output_var_mask[:, :, None].expand(batch_size, num_out_vars, h * w).reshape(batch_size, out_len)

        if output_var_ids is None:
            output_var_ids = torch.arange(num_out_vars, dtype=torch.long, device=embeds.device).unsqueeze(0).expand(batch_size, -1)
        output_var_ids = output_var_ids.clamp(min=0, max=self.output_var_embed.num_embeddings - 1)
        output_var_flat = output_var_ids[:, :, None].expand(batch_size, num_out_vars, h * w).reshape(batch_size, out_len)

        dec_x = self.output_token.repeat(batch_size, out_len, 1)
        dec_x = dec_x + self.output_var_embed(output_var_flat) + cond_emb.unsqueeze(1)
        out_spatial = torch.arange(h * w, device=embeds.device, dtype=torch.long).repeat(num_out_vars)
        out_spatial = out_spatial.unsqueeze(0).expand(batch_size, -1)

        if self.token_type == "phaedra":
            amp_hidden_states = dec_x
            for blk in self.amplitude_decoder:
                amp_hidden_states = blk(
                    amp_hidden_states,
                    memory=memory,
                    tgt_mask=dec_valid,
                    memory_mask=enc_mask,
                    tgt_spatial_indices=out_spatial,
                )

            amp_logits = self.amp_head(amp_hidden_states)

            # memory: [B, S_enc, D], amp_hidden_states: [B, S_out, D]
            # morph_memory: [B, S_enc + S_out, D]
            morph_memory = torch.cat([memory, amp_hidden_states], dim=1)
            # enc_mask: [B, S_enc], dec_valid: [B, S_out]
            # morph_memory_mask: [B, S_enc + S_out] to align with morph_memory sequence length.
            morph_memory_mask = torch.cat([enc_mask, dec_valid], dim=1)

            morph_hidden_states = dec_x
            for blk in self.morphology_decoder:
                morph_hidden_states = blk(
                    morph_hidden_states,
                    memory=morph_memory,
                    tgt_mask=dec_valid,
                    memory_mask=morph_memory_mask,
                    tgt_spatial_indices=out_spatial,
                )

            logits_morph = self.morph_head(morph_hidden_states)
            return amp_logits, logits_morph

        hidden_states = dec_x
        for blk in self.decoder:
            hidden_states = blk(
                hidden_states,
                memory=memory,
                tgt_mask=dec_valid,
                memory_mask=enc_mask,
                tgt_spatial_indices=out_spatial,
            )

        fsq_logits = self.fsq_head(hidden_states)
        return fsq_logits

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
    ):
        if self.token_type == "phaedra":
            amp_logits, logits_morph = self.forward(
                input_amp=input_amp,
                input_morph=input_morph,
                input_time_idx=input_time_idx,
                output_time_idx=output_time_idx,
                input_var_ids=input_var_ids,
                output_var_ids=output_var_ids,
                input_var_mask=input_var_mask,
                output_var_mask=output_var_mask,
                problem_type_id=problem_type_id,
            )
            pred_amp = amp_logits.argmax(dim=-1)
            pred_morph = logits_morph.argmax(dim=-1)
            return pred_amp, pred_morph

        fsq_logits = self.forward(
            input_tokens=input_tokens,
            input_time_idx=input_time_idx,
            output_time_idx=output_time_idx,
            input_var_ids=input_var_ids,
            output_var_ids=output_var_ids,
            input_var_mask=input_var_mask,
            output_var_mask=output_var_mask,
            problem_type_id=problem_type_id,
        )
        return fsq_logits.argmax(dim=-1)
