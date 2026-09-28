"""Seq2seq-compatible attention blocks for the benchmark variants.

These mirror `hub.models.seq2seq_model.RoPEEncoderBlock` / `.RoPEDecoderBlock`
exactly in interface (kwargs: `attention_mask`, `tgt_mask`, `memory_mask`,
`spatial_indices`, `tgt_spatial_indices`) and parameter shape, so a subclass of
`OperatorLearningModel` can swap them in for the stock Flash-SDPA blocks
without any other changes.

- LinearAttn*Block: kernel feature-map linear attention with phi(x)=elu(x)+1
  (Katharopoulos et al. 2020).
- LongformerEncoderBlock: sliding-window self-attention via
  `torch.nn.attention.flex_attention` with optional leading global tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hub.models.rope_utils import apply_2d_rope

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    _HAVE_FLEX = True
except Exception:  # pragma: no cover
    _HAVE_FLEX = False


# Module-level pre-compiled flex_attention. Sharing a single compiled wrapper
# across all LongformerEncoderBlock instances ensures:
#   1. Calls inside an outer `torch.compile(model)` graph are traced by dynamo
#      as the flex_attention HOP (fused Triton kernel, no [B,H,L,L]
#      materialisation, no warning).
#   2. Calls in pure eager mode (no outer compile) still go through this
#      wrapper, so the inner compile activates and the warning
#      "flex_attention called without torch.compile()" never fires.
# This must NOT be created lazily inside `forward` -- doing so causes a graph
# break under the outer compile, which is exactly what was happening before.
_COMPILED_FLEX = torch.compile(flex_attention, dynamic=False) if _HAVE_FLEX else None


# ---------------------------------------------------------------------------
# Mask-free Flash SDPA blocks.
#
# Identical to `hub.models.seq2seq_model.RoPE{Encoder,Decoder}Block` in every
# respect (same Linear/LayerNorm/MLP shape -> same parameter count) EXCEPT
# they pass `attn_mask=None` to `F.scaled_dot_product_attention`. The parent
# blocks pass a `[B, 1, 1, L]` bool mask even when every position is valid;
# under torch.compile that mask trips SDPA off the Flash backend onto the
# math backend, which materialises the full `[B*H, L_q, L_k]` fp32 attention
# matrix (16 GiB at B=32, H=8, L=4096) and OOMs.
#
# Our seq2seq variants always have all V variables active and every spatial
# position valid, so the mask is logically a no-op and these blocks are
# correct drop-in replacements.
# ---------------------------------------------------------------------------
class FlashEncoderBlock(nn.Module):
    """RoPE encoder block; no SDPA mask, no output masking."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, grid_size: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_width = grid_size

        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

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
        del attention_mask  # assumed all-True for our benchmark datasets
        h = self.norm1(x)
        q = self._shape(self.q_proj(h))
        k = self._shape(self.k_proj(h))
        v = self._shape(self.v_proj(h))
        q, k = apply_2d_rope(q, k, spatial_indices=spatial_indices, grid_width=self.grid_width)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(x.shape)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class FlashDecoderBlock(nn.Module):
    """RoPE decoder block; no SDPA masks (self or cross), no output masking."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, grid_size: int):
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

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor,
                memory_mask: torch.Tensor, tgt_spatial_indices: torch.Tensor) -> torch.Tensor:
        del tgt_mask, memory_mask  # assumed all-True
        h = self.norm1(x)
        q = self._shape(self.self_q(h))
        k = self._shape(self.self_k(h))
        v = self._shape(self.self_v(h))
        q, k = apply_2d_rope(q, k, spatial_indices=tgt_spatial_indices, grid_width=self.grid_width)
        self_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        self_out = self_out.transpose(1, 2).reshape(x.shape)
        x = x + self.self_out(self_out)

        h = self.norm2(x)
        q = self._shape(self.cross_q(h))
        k = self._shape(self.cross_k(memory))
        v = self._shape(self.cross_v(memory))
        cross_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        cross_out = cross_out.transpose(1, 2).reshape(x.shape)
        x = x + self.cross_out(cross_out)

        x = x + self.mlp(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Linear attention (Katharopoulos et al. 2020).
# ---------------------------------------------------------------------------
def _phi(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def _linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      kv_mask: torch.Tensor | None) -> torch.Tensor:
    """q, k, v: [B, H, L, D]. kv_mask: [B, L_k] bool (or None)."""
    q = _phi(q)
    k = _phi(k)
    if kv_mask is not None:
        m = kv_mask[:, None, :, None].to(k.dtype)
        k = k * m
        v = v * m
    kv = torch.einsum("bhnd,bhne->bhde", k, v)                    # [B, H, D, D]
    k_sum = k.sum(dim=2)                                           # [B, H, D]
    num = torch.einsum("bhnd,bhde->bhne", q, kv)                  # [B, H, L_q, D]
    den = torch.einsum("bhnd,bhd->bhn", q, k_sum).unsqueeze(-1)   # [B, H, L_q, 1]
    return num / den.clamp(min=1e-6)


class LinearAttnEncoderBlock(nn.Module):
    """Encoder block with the seq2seq parameter names; linear attention."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, grid_size: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_width = grid_size

        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

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
        q = self._shape(self.q_proj(h))
        k = self._shape(self.k_proj(h))
        v = self._shape(self.v_proj(h))
        q, k = apply_2d_rope(q, k, spatial_indices=spatial_indices, grid_width=self.grid_width)
        attn = _linear_attention(q, k, v, kv_mask=attention_mask.to(torch.bool))
        attn = attn.transpose(1, 2).reshape(x.shape)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)


class LinearAttnDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, grid_size: int):
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

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor,
                memory_mask: torch.Tensor, tgt_spatial_indices: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q = self._shape(self.self_q(h))
        k = self._shape(self.self_k(h))
        v = self._shape(self.self_v(h))
        q, k = apply_2d_rope(q, k, spatial_indices=tgt_spatial_indices, grid_width=self.grid_width)
        self_out = _linear_attention(q, k, v, kv_mask=tgt_mask.to(torch.bool))
        self_out = self_out.transpose(1, 2).reshape(x.shape)
        x = x + self.self_out(self_out)

        h = self.norm2(x)
        q = self._shape(self.cross_q(h))
        k = self._shape(self.cross_k(memory))
        v = self._shape(self.cross_v(memory))
        cross_out = _linear_attention(q, k, v, kv_mask=memory_mask.to(torch.bool))
        cross_out = cross_out.transpose(1, 2).reshape(x.shape)
        x = x + self.cross_out(cross_out)

        x = x + self.mlp(self.norm3(x))
        return x * tgt_mask.unsqueeze(-1).to(x.dtype)


# ---------------------------------------------------------------------------
# Longformer sliding-window encoder block.
# ---------------------------------------------------------------------------
def _sliding_window_mask_mod(window_size: int, num_global: int):
    def _mod(b, h, q_idx, kv_idx):
        within = (q_idx - kv_idx).abs() <= window_size
        global_q = q_idx < num_global
        global_kv = kv_idx < num_global
        return within | global_q | global_kv
    return _mod


class _DenseSlidingWindow(nn.Module):
    """CPU fallback: build [L, L] mask once and call SDPA."""

    def __init__(self, window_size: int, num_global: int, seq_len: int):
        super().__init__()
        i = torch.arange(seq_len)
        within = (i[:, None] - i[None, :]).abs() <= window_size
        glob = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        glob[:num_global, :] = True
        glob[:, :num_global] = True
        self.register_buffer("mask", (within | glob), persistent=False)

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask[None, None, :, :], dropout_p=0.0,
        )


class LongformerEncoderBlock(nn.Module):
    """Sliding-window self-attention encoder block in the seq2seq style.

    `seq_len` is the encoder sequence length; for the seq2seq backbone this is
    V * H * W = 4096 (4 vars on a 32x32 grid). flex_attention's block mask is
    built lazily on first CUDA forward so the block can be moved freely between
    CPU (dense fallback) and GPU (compiled kernel).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, grid_size: int,
                 seq_len: int, window_size: int, num_global: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_width = grid_size
        self.seq_len = seq_len
        self.window_size = window_size
        self.num_global = num_global

        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

        self._block_mask = None
        self._dense_fallback = _DenseSlidingWindow(window_size, num_global, seq_len)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def setup_block_mask(self, device: torch.device) -> None:
        """Build the sliding-window BlockMask EAGERLY on the given device.

        Must be called after `.to(device)` and BEFORE `torch.compile(model)` --
        if the BlockMask is created lazily inside forward, dynamo treats it as
        a side effect on `self`, graph-breaks, and the subsequent
        `flex_attention` call falls into the eager (unfused, [B,H,L,L]
        materialising) path with a warning.

        On non-CUDA devices we leave `self._block_mask=None` and use the dense
        SDPA fallback at forward time.
        """
        if not _HAVE_FLEX or device.type != "cuda":
            return
        if self._block_mask is not None:
            return
        mod = _sliding_window_mask_mod(self.window_size, self.num_global)
        self._block_mask = create_block_mask(
            mod, B=None, H=None, Q_LEN=self.seq_len, KV_LEN=self.seq_len, device=device,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        q = self._shape(self.q_proj(h))
        k = self._shape(self.k_proj(h))
        v = self._shape(self.v_proj(h))
        q, k = apply_2d_rope(q, k, spatial_indices=spatial_indices, grid_width=self.grid_width)

        if _COMPILED_FLEX is not None and self._block_mask is not None:
            attn = _COMPILED_FLEX(q, k, v, block_mask=self._block_mask)
        else:
            attn = self._dense_fallback(q, k, v)

        attn = attn.transpose(1, 2).reshape(x.shape)
        x = x + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x * attention_mask.unsqueeze(-1).to(x.dtype)
