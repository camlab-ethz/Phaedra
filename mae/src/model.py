from __future__ import annotations

import math
from dataclasses import dataclass
import os

import torch
import torch.nn as nn


@dataclass
class MAEConfig:
    token_type: str
    vocab_amp: int | None
    vocab_morph: int | None
    vocab_fsq: int | None
    num_vars: int
    grid_size: int
    embed_dim: int
    encoder_depth: int
    decoder_depth: int
    num_heads: int
    mlp_ratio: int
    mask_ratio: float
    fusion: str
    num_datasets: int = 1


def _build_transformer(depth: int, embed_dim: int, num_heads: int, mlp_ratio: int) -> nn.Module:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=embed_dim * mlp_ratio,
        batch_first=True,
        activation="gelu",
        norm_first=True,  # Recommended for MAE stability
    )
    return nn.TransformerEncoder(
        encoder_layer, 
        num_layers=depth,
        enable_nested_tensor=False  # THIS FIXES THE CRASH
    )

def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    bad = (~torch.isfinite(tensor)).sum().item()
    raise ValueError(f"Non-finite values in {name}: {bad} of {tensor.numel()}")

def get_sincos_2d_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Generates fixed 2D sinusoidal positional embeddings for a grid."""
    if embed_dim % 4 != 0:
        raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos embeddings")

    half_dim = embed_dim // 2
    freq_dim = half_dim // 2
    div_term = torch.exp(torch.arange(0, freq_dim, dtype=torch.float32) * (-math.log(10000.0) / freq_dim))

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = grid_x.reshape(-1, 1)
    grid_y = grid_y.reshape(-1, 1)

    emb_x = torch.cat([torch.sin(grid_x * div_term), torch.cos(grid_x * div_term)], dim=1)
    emb_y = torch.cat([torch.sin(grid_y * div_term), torch.cos(grid_y * div_term)], dim=1)
    return torch.cat([emb_x, emb_y], dim=1)

class TokenMAE(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg
        self.grid_size = cfg.grid_size
        self.num_vars = cfg.num_vars
        self.mask_ratio = cfg.mask_ratio
        self.fusion = cfg.fusion
        if cfg.num_datasets < 1:
            raise ValueError("num_datasets must be >= 1")

        if cfg.token_type == "phaedra":
            if cfg.vocab_amp is None or cfg.vocab_morph is None:
                raise ValueError("Phaedra vocab sizes required")
            self.embed_amp = nn.Embedding(cfg.vocab_amp, cfg.embed_dim)
            self.embed_morph = nn.Embedding(cfg.vocab_morph, cfg.embed_dim)
        else:
            if cfg.vocab_fsq is None:
                raise ValueError("FSQ vocab size required")
            self.embed_fsq = nn.Embedding(cfg.vocab_fsq, cfg.embed_dim)

        if cfg.fusion == "concat":
            self.fusion_proj = nn.Linear(cfg.embed_dim * 2, cfg.embed_dim)
        else:
            self.fusion_proj = None

        self.var_embed = nn.Embedding(cfg.num_vars, cfg.embed_dim)
        self.dataset_embed = nn.Embedding(cfg.num_datasets, cfg.embed_dim)
        nn.init.trunc_normal_(self.dataset_embed.weight, std=0.02)
        # self.pos_embed = nn.Parameter(torch.zeros(cfg.grid_size * cfg.grid_size, cfg.embed_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Replace it with a fixed, non-trainable buffer:
        self.register_buffer(
            "pos_embed",
            get_sincos_2d_pos_embed(cfg.embed_dim, cfg.grid_size),
        )

        self.encoder = _build_transformer(cfg.encoder_depth, cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
        self.decoder = _build_transformer(cfg.decoder_depth, cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio)
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        if cfg.token_type == "phaedra":
            self.head_amp = nn.Linear(cfg.embed_dim, 1)
            self.head_morph = nn.Linear(cfg.embed_dim, cfg.vocab_morph)
        else:
            self.head_fsq = nn.Linear(cfg.embed_dim, cfg.vocab_fsq)

    def _make_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        total_tokens = self.grid_size * self.grid_size
        num_mask = int(total_tokens * self.mask_ratio)
        
        # Generate random noise for the whole batch at once
        noise = torch.rand(batch_size, total_tokens, device=device)
        
        # Sort the noise to get random permutation indices per row
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Sort the indices to get the "rank" of each original token
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Tokens with a rank lower than num_mask are masked (True)
        mask = ids_restore < num_mask
        
        return mask

    def _embed_tokens(self, tokens_amp: torch.Tensor | None, tokens_morph: torch.Tensor | None, tokens_fsq: torch.Tensor | None) -> torch.Tensor:
        if self.cfg.token_type == "phaedra":
            amp_emb = self.embed_amp(tokens_amp)
            morph_emb = self.embed_morph(tokens_morph)
            if self.fusion == "concat":
                merged = torch.cat([amp_emb, morph_emb], dim=-1)
                return self.fusion_proj(merged)
            return amp_emb + morph_emb
        return self.embed_fsq(tokens_fsq)

    def _resolve_dataset_ids(
        self,
        dataset_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if dataset_ids is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if dataset_ids.ndim > 1:
            dataset_ids = dataset_ids.view(-1)
        if dataset_ids.shape[0] != batch_size:
            raise ValueError(
                f"dataset_ids must have shape [batch_size]; got {tuple(dataset_ids.shape)} for batch_size={batch_size}"
            )

        dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
        if dataset_ids.numel() > 0:
            ds_min = int(dataset_ids.min().item())
            ds_max = int(dataset_ids.max().item())
            if ds_min < 0 or ds_max >= self.cfg.num_datasets:
                raise ValueError(
                    f"dataset_ids out of range [0, {self.cfg.num_datasets - 1}]: min={ds_min}, max={ds_max}"
                )
        return dataset_ids

    def forward(self, tokens_amp=None, tokens_morph=None, tokens_fsq=None, dataset_ids: torch.Tensor | None = None):
        debug_nan = os.getenv("MAE_DEBUG_NAN") == "1"
        device = tokens_amp.device if tokens_amp is not None else tokens_fsq.device
        batch_size, num_vars, h, w = (tokens_amp.shape if tokens_amp is not None else tokens_fsq.shape)
        tokens_amp = tokens_amp.reshape(batch_size, num_vars, h * w) if tokens_amp is not None else None
        tokens_morph = tokens_morph.reshape(batch_size, num_vars, h * w) if tokens_morph is not None else None
        tokens_fsq = tokens_fsq.reshape(batch_size, num_vars, h * w) if tokens_fsq is not None else None

        mask = self._make_mask(batch_size, device)

        embeds = self._embed_tokens(tokens_amp, tokens_morph, tokens_fsq)
        embeds = embeds.view(batch_size, num_vars * h * w, -1)
        if debug_nan:
            _assert_finite("embeds", embeds)

        var_ids = torch.arange(num_vars, device=device).repeat_interleave(h * w)
        var_emb = self.var_embed(var_ids)[None, :, :]
        pos_emb = self.pos_embed[None, None, :, :].expand(1, num_vars, -1, -1).reshape(1, num_vars * h * w, -1)
        dataset_ids = self._resolve_dataset_ids(dataset_ids, batch_size, device)
        dataset_emb = self.dataset_embed(dataset_ids)[:, None, :]
        var_emb = var_emb.to(dtype=embeds.dtype)
        pos_emb = pos_emb.to(dtype=embeds.dtype)
        dataset_emb = dataset_emb.to(dtype=embeds.dtype)
        embeds = embeds + var_emb + pos_emb + dataset_emb
        if debug_nan:
            _assert_finite("embeds+pos", embeds)

        mask_full = mask.unsqueeze(1).repeat(1, num_vars, 1).view(batch_size, num_vars * h * w)
        visible_indices = ~mask_full

        visible_counts = visible_indices.sum(dim=1)
        if torch.any(visible_counts != visible_counts[0]):
            raise ValueError("Visible token counts differ across batch; masking should be uniform")

        vis_count = int(visible_counts[0].item())
        visible_idx = visible_indices.nonzero(as_tuple=False)
        visible_idx = visible_idx[:, 1].view(batch_size, vis_count)
        visible_tokens = embeds.gather(1, visible_idx.unsqueeze(-1).expand(-1, -1, embeds.size(-1)))
        encoder_out = self.encoder(visible_tokens)
        if debug_nan:
            _assert_finite("encoder_out", encoder_out)
        encoder_out = self.decoder_embed(encoder_out)
        full_dtype = encoder_out.dtype

        full_tokens = self.mask_token.to(dtype=full_dtype).repeat(batch_size, num_vars * h * w, 1)
        full_tokens.scatter_(1, visible_idx.unsqueeze(-1).expand(-1, -1, self.cfg.embed_dim), encoder_out)

        # Add positional and variable embeddings before decoding, but only to masked positions.
        combined_emb = (var_emb + pos_emb + dataset_emb).to(dtype=full_dtype)
        full_tokens = full_tokens + combined_emb
        decoded = self.decoder(full_tokens)
        if debug_nan:
            _assert_finite("decoded", decoded)

        if self.cfg.token_type == "phaedra":
            amp_pred = self.head_amp(decoded).squeeze(-1)
            logits_morph = self.head_morph(decoded)
            if debug_nan:
                _assert_finite("amp_pred", amp_pred)
                _assert_finite("logits_morph", logits_morph)
            return amp_pred, logits_morph, mask_full
        logits = self.head_fsq(decoded)
        if debug_nan:
            _assert_finite("logits", logits)
        return logits, mask_full

    def predict(self, tokens_amp=None, tokens_morph=None, tokens_fsq=None, dataset_ids: torch.Tensor | None = None):
        if self.cfg.token_type == "phaedra":
            amp_pred, logits_morph, mask_full = self.forward(tokens_amp, tokens_morph, None, dataset_ids=dataset_ids)
            pred_amp = amp_pred.round().clamp(0, self.cfg.vocab_amp - 1).long()
            pred_morph = logits_morph.argmax(dim=-1)
            return pred_amp, pred_morph, mask_full
        logits, mask_full = self.forward(None, None, tokens_fsq, dataset_ids=dataset_ids)
        pred = logits.argmax(dim=-1)
        return pred, mask_full
