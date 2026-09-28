"""Continuous-latent seq2seq operator (baseline of the main comparison).

Same skeleton as the paper's token transformer, with the discrete parts
swapped for continuous ones — nothing else changes:

  tokens + embeddings + CE heads     ->  latents + Linear(8->d) + Linear(d->8), L1
  bifurcated amp/morph decoders      ->  single decoder (regression needs no split)
  masked SDPA                        ->  mask-free SDPA (FlashAttention-2 kernel)

Reuses the battle-tested mask-free FA2 blocks from baselines.arch
(FlashEncoderBlock / FlashDecoderBlock: pre-norm, 2D RoPE on self-attn,
identical parameterization to the hub blocks).

Sequence layout identical to the token models: 4096 = V(4) x 32 x 32 positions,
per-position feature = that variable's 8-dim latent vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from baselines.arch.attention_blocks import (
    FlashDecoderBlock,
    FlashEncoderBlock,
)


@dataclass
class ContinuousOperatorConfig:
    latent_channels: int = 8
    num_vars: int = 4
    grid_size: int = 32
    embed_dim: int = 320
    encoder_depth: int = 19
    decoder_depth: int = 9
    num_heads: int = 8
    mlp_ratio: int = 4
    max_time_index: int = 14
    grad_checkpointing: bool = True


class ContinuousLatentOperator(nn.Module):
    def __init__(self, cfg: ContinuousOperatorConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.embed_dim)
        if (d // cfg.num_heads) % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")

        self.in_proj = nn.Linear(int(cfg.latent_channels), d)
        self.var_embed = nn.Embedding(int(cfg.num_vars), d)
        self.lead_time_embed = nn.Embedding(int(cfg.max_time_index) + 1, d)
        self.output_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.output_token, std=0.02)

        self.encoder = nn.ModuleList([
            FlashEncoderBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
            for _ in range(int(cfg.encoder_depth))
        ])
        self.decoder = nn.ModuleList([
            FlashDecoderBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.grid_size)
            for _ in range(int(cfg.decoder_depth))
        ])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, int(cfg.latent_channels))

    def _flatten(self, z: torch.Tensor) -> torch.Tensor:
        # [B, V, C, H, W] -> [B, V*H*W, C]
        B, V, C, H, W = z.shape
        return z.permute(0, 1, 3, 4, 2).reshape(B, V * H * W, C)

    def forward(
        self,
        input_latents: torch.Tensor,     # [B, V, C, H, W] float
        input_time_idx: torch.Tensor,    # [B] long
        output_time_idx: torch.Tensor,   # [B] long
    ) -> torch.Tensor:                    # [B, V, C, H, W]
        cfg = self.cfg
        B, V, C, H, W = input_latents.shape
        device = input_latents.device
        L = V * H * W

        delta_t = (output_time_idx.long() - input_time_idx.long()).clamp(
            min=0, max=cfg.max_time_index)
        cond = self.lead_time_embed(delta_t)                       # [B, d]

        var_ids = torch.arange(V, device=device).repeat_interleave(H * W)  # [L]
        spatial = torch.arange(H * W, device=device, dtype=torch.long).repeat(V)
        spatial = spatial.unsqueeze(0).expand(B, -1)
        mask = torch.ones((B, L), dtype=torch.bool, device=device)

        x = self.in_proj(self._flatten(input_latents))             # [B, L, d]
        x = x + self.var_embed(var_ids)[None] + cond[:, None, :]
        gc = bool(self.cfg.grad_checkpointing) and self.training
        for blk in self.encoder:
            if gc:
                x = checkpoint(lambda t, b=blk: b(t, attention_mask=mask,
                                                  spatial_indices=spatial),
                               x, use_reentrant=False)
            else:
                x = blk(x, attention_mask=mask, spatial_indices=spatial)

        q = self.output_token.expand(B, L, -1)
        q = q + self.var_embed(var_ids)[None] + cond[:, None, :]
        for blk in self.decoder:
            if gc:
                q = checkpoint(lambda t, m, b=blk: b(t, memory=m, tgt_mask=mask,
                                                     memory_mask=mask,
                                                     tgt_spatial_indices=spatial),
                               q, x, use_reentrant=False)
            else:
                q = blk(q, memory=x, tgt_mask=mask, memory_mask=mask,
                        tgt_spatial_indices=spatial)
        out = self.out_proj(self.out_norm(q))                      # [B, L, C]
        return out.view(B, V, H, W, C).permute(0, 1, 4, 2, 3).contiguous()

    @torch.no_grad()
    def predict(self, input_latents, input_time_idx, output_time_idx):
        return self.forward(input_latents, input_time_idx, output_time_idx)


def build_continuous_operator(cfg: dict) -> ContinuousLatentOperator:
    m = cfg["model"]
    return ContinuousLatentOperator(ContinuousOperatorConfig(
        latent_channels=int(m.get("latent_channels", 8)),
        num_vars=int(m.get("num_vars", 4)),
        grid_size=int(m.get("grid_size", 32)),
        embed_dim=int(m["embed_dim"]),
        encoder_depth=int(m["encoder_depth"]),
        decoder_depth=int(m["decoder_depth"]),
        num_heads=int(m.get("num_heads", 8)),
        mlp_ratio=int(m.get("mlp_ratio", 4)),
        max_time_index=int(m.get("max_time_index", 14)),
        grad_checkpointing=bool(m.get("grad_checkpointing", True)),
    ))
