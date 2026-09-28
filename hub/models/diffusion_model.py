from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope_utils import apply_2d_rope


@dataclass
class DiffusionTransformerConfig:
    morph_vocab_size: int
    amp_vocab_size: int
    num_variable_types: int
    variable_pad_id: int
    grid_height: int
    grid_width: int
    max_lead_time: int
    diffusion_steps: int
    hidden_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    rope_base: float = 10000.0
    token_embedding_init_std: float = 0.02
    aux_embedding_init_std: float = 0.005
    aux_embedding_scale_init: float = 0.25
    layerscale: bool = False
    layerscale_init: float = 1e-3


def timestep_sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / max(1, half)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RoPETransformerBlock(nn.Module):
    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (cfg.hidden_dim // cfg.num_heads) % 4 != 0:
            raise ValueError("(hidden_dim // num_heads) must be divisible by 4 for 2D RoPE")

        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.grid_width = cfg.grid_width
        self.rope_base = cfg.rope_base
        self.dropout = cfg.dropout

        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * cfg.mlp_ratio),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * cfg.mlp_ratio, cfg.hidden_dim),
            nn.Dropout(cfg.dropout),
        )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)

        q = self._shape(self.q_proj(h))
        k = self._shape(self.k_proj(h))
        v = self._shape(self.v_proj(h))

        q, k = apply_2d_rope(
            q=q,
            k=k,
            spatial_indices=spatial_indices,
            grid_width=self.grid_width,
            base=self.rope_base,
        )

        key_padding_mask = attention_mask.to(torch.bool)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=key_padding_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_out = attn_out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        x = residual + self.out_proj(attn_out)

        x = x + self.mlp(self.norm2(x))
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x


class MaskedPDEDiffusionTransformer(nn.Module):
    def __init__(self, cfg: DiffusionTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.morph_embedding = nn.Embedding(cfg.morph_vocab_size, cfg.hidden_dim)
        self.amp_embedding = nn.Embedding(cfg.amp_vocab_size, cfg.hidden_dim)
        self.variable_embedding = nn.Embedding(cfg.num_variable_types + 1, cfg.hidden_dim)
        self.segment_embedding = nn.Embedding(2, cfg.hidden_dim)
        self.pde_type_embedding = nn.Embedding(2, cfg.hidden_dim)
        self.lead_time_embedding = nn.Embedding(cfg.max_lead_time + 1, cfg.hidden_dim)

        self.variable_scale = nn.Parameter(torch.tensor(cfg.aux_embedding_scale_init))
        self.segment_scale = nn.Parameter(torch.tensor(cfg.aux_embedding_scale_init))
        self.pde_type_scale = nn.Parameter(torch.tensor(cfg.aux_embedding_scale_init))
        self.lead_time_scale = nn.Parameter(torch.tensor(cfg.aux_embedding_scale_init))
        self.time_scale = nn.Parameter(torch.tensor(cfg.aux_embedding_scale_init))

        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.blocks = nn.ModuleList([RoPETransformerBlock(cfg) for _ in range(cfg.depth)])
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

        self.morph_head = nn.Linear(cfg.hidden_dim, cfg.morph_vocab_size)
        self.amp_head = nn.Linear(cfg.hidden_dim, cfg.amp_vocab_size)
        self._init_parameters()

    def _init_parameters(self) -> None:
        token_std = self.cfg.token_embedding_init_std
        aux_std = self.cfg.aux_embedding_init_std

        nn.init.normal_(self.morph_embedding.weight, mean=0.0, std=token_std)
        nn.init.normal_(self.amp_embedding.weight, mean=0.0, std=token_std)
        nn.init.normal_(self.variable_embedding.weight, mean=0.0, std=aux_std)
        nn.init.normal_(self.segment_embedding.weight, mean=0.0, std=aux_std)
        nn.init.normal_(self.pde_type_embedding.weight, mean=0.0, std=aux_std)
        nn.init.normal_(self.lead_time_embedding.weight, mean=0.0, std=aux_std)

        if 0 <= self.cfg.variable_pad_id < self.variable_embedding.num_embeddings:
            with torch.no_grad():
                self.variable_embedding.weight[self.cfg.variable_pad_id].zero_()

        for module in self.diffusion_time_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        morph_tokens: torch.Tensor,
        amp_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        variable_ids: torch.Tensor,
        spatial_indices: torch.Tensor,
        lead_time: torch.Tensor,
        pde_type_id: torch.Tensor,
        diffusion_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if morph_tokens.shape != amp_tokens.shape:
            raise ValueError("morph_tokens and amp_tokens must have the same shape")

        bsz, seq_len = morph_tokens.shape
        if attention_mask.shape != (bsz, seq_len):
            raise ValueError("attention_mask shape mismatch")
        if segment_ids.shape != (bsz, seq_len):
            raise ValueError("segment_ids shape mismatch")
        if variable_ids.shape != (bsz, seq_len):
            raise ValueError("variable_ids shape mismatch")
        if spatial_indices.shape != (bsz, seq_len):
            raise ValueError("spatial_indices shape mismatch")

        valid_spatial = spatial_indices[spatial_indices >= 0]
        if valid_spatial.numel() > 0:
            max_valid = int(valid_spatial.max().item())
            max_allowed = int(self.cfg.grid_height * self.cfg.grid_width - 1)
            if max_valid > max_allowed:
                raise ValueError(
                    f"spatial_indices max={max_valid} exceeds configured grid range [0, {max_allowed}]"
                )

        token_embed = self.morph_embedding(morph_tokens) + self.amp_embedding(amp_tokens)
        var_embed = self.variable_scale * self.variable_embedding(variable_ids)
        seg_embed = self.segment_scale * self.segment_embedding(segment_ids)

        lead_time = lead_time.clamp(min=0, max=self.cfg.max_lead_time)
        lead_embed = self.lead_time_scale * self.lead_time_embedding(lead_time).unsqueeze(1)
        pde_embed = self.pde_type_scale * self.pde_type_embedding(pde_type_id.clamp(min=0, max=1)).unsqueeze(1)

        time_embed = timestep_sinusoidal_embedding(diffusion_timestep, self.cfg.hidden_dim)
        time_embed = self.time_scale * self.diffusion_time_mlp(time_embed).unsqueeze(1)

        x = token_embed + var_embed + seg_embed + lead_embed + pde_embed + time_embed
        attn_mask = attention_mask
        spat = spatial_indices

        x = x * attn_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            x = block(x, attention_mask=attn_mask, spatial_indices=spat)

        x = self.final_norm(x)
        x = x * attn_mask.unsqueeze(-1).to(x.dtype)

        morph_logits = self.morph_head(x)
        amp_logits = self.amp_head(x)
        return morph_logits, amp_logits
