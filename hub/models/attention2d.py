"""Pluggable 2D self-attention variants for the seq2seq encoder.

The seq2seq encoder's self-attention is the locus of the "how do we mix tokens
to capture structure" question. In 2D the full-attention path (RoPEEncoderBlock's
inline SDPA) is affordable (~4k tokens) and serves as the ORACLE; these modules
are the sub-quadratic approximations we benchmark against it:

  full     -> stays inline in RoPEEncoderBlock (oracle; this file not used)
  linear   -> elu+1 linear attention, O(N), full receptive field (implemented)
  windowed -> non-overlapping 2D windows (local)            [phase 2]
  ssm      -> 2D bi-axial selective state-space             [phase 3]
  radial   -> static distance-decaying sparse softmax       [phase 4]
  quadtree -> hierarchical coarse-to-fine top-K softmax     [phase 5]

Each module owns its own q/k/v/out projections and applies 2D RoPE, so swapping
the mode keeps everything else (embeddings, decoder, heads) identical — a clean
apples-to-apples comparison. Interface mirrors the inline full path:
    forward(x_normed[b,L,D], attention_mask[b,L], spatial_indices[b,L]) -> [b,L,D]
returning the post-out_proj result to be added as the residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .rope_utils import apply_2d_rope
# `SelectiveSSM1D` (3D state-space variant) is not part of this release; the
# 'ssm' attention kind imports it lazily so the 2D models never need it.


class LinearSelfAttention2D(nn.Module):
    """elu+1 linear attention: O(N) with a full (global) receptive field.

    out_i = phi(q_i) . (sum_j phi(k_j) v_j^T) / (phi(q_i) . sum_j phi(k_j)),
    with invalid (masked) tokens zeroed out of the key/value sums. 2D RoPE is
    applied to q/k for positional structure, consistent with the full-attention
    oracle. Sums are accumulated in fp32 for stability.
    """

    def __init__(self, embed_dim: int, num_heads: int, grid_size: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid_width = grid_size
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        q, k = apply_2d_rope(q, k, spatial_indices=spatial_indices, grid_width=self.grid_width)
        qf = (F.elu(q) + 1.0).float()
        kf = (F.elu(k) + 1.0).float()
        vf = v.float()
        m = attention_mask[:, None, :, None].to(vf.dtype)  # [b,1,L,1]
        kf = kf * m
        vf = vf * m
        kv = torch.einsum("bhld,bhle->bhde", kf, vf)          # [b,H,hd,hd]
        z = kf.sum(dim=2)                                      # [b,H,hd]
        num = torch.einsum("bhld,bhde->bhle", qf, kv)         # [b,H,L,hd]
        den = torch.einsum("bhld,bhd->bhl", qf, z).clamp_min(1e-6).unsqueeze(-1)
        out = (num / den).to(x.dtype)
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.out_proj(out)


class WindowedSelfAttention2D(nn.Module):
    """Non-overlapping w x w window attention within each field's HxW grid (local
    receptive field). Softmax SDPA per window, 2D RoPE with absolute coordinates."""

    def __init__(self, embed_dim: int, num_heads: int, grid_size: int, window: int = 8) -> None:
        super().__init__()
        if grid_size % window != 0:
            raise ValueError(f"window {window} must divide grid_size {grid_size}")
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid = grid_size
        self.window = window
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        g, w, n = grid_size, window, grid_size // window
        idx = torch.arange(g * g).view(n, w, n, w).permute(0, 2, 1, 3).reshape(n * n, w * w)
        self.register_buffer("win_spatial", idx, persistent=False)  # [nW, w*w] absolute flat coords

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        b, L, d = x.shape
        g, w = self.grid, self.window
        n = g // w
        nW = n * n
        V = L // (g * g)
        H, hd = self.num_heads, self.head_dim

        def part(t):  # [b,L,d] -> [b*V*nW, w*w, d]
            return t.view(b, V, n, w, n, w, d).permute(0, 1, 2, 4, 3, 5, 6).reshape(b * V * nW, w * w, d)

        qw, kw, vw = part(self.q_proj(x)), part(self.k_proj(x)), part(self.v_proj(x))
        m = qw.shape[0]
        qh = qw.view(m, w * w, H, hd).transpose(1, 2)
        kh = kw.view(m, w * w, H, hd).transpose(1, 2)
        vh = vw.view(m, w * w, H, hd).transpose(1, 2)
        sp = self.win_spatial.unsqueeze(0).expand(b * V, nW, w * w).reshape(m, w * w)
        qh, kh = apply_2d_rope(qh, kh, spatial_indices=sp, grid_width=g)
        out = F.scaled_dot_product_attention(qh, kh, vh)
        out = out.transpose(1, 2).reshape(m, w * w, d)
        out = out.view(b, V, n, n, w, w, d).permute(0, 1, 2, 4, 3, 5, 6).reshape(b, V * g * g, d)
        return self.out_proj(out)


class SSM2DSelfAttention(nn.Module):
    """Bidirectional selective state-space mixing along the two spatial axes (full
    receptive field over depth). Reuses the validated 3D SelectiveSSM1D; scan order
    encodes position, so no RoPE. num_heads is unused (SSM is not multi-head)."""

    def __init__(self, embed_dim: int, num_heads: int, grid_size: int, d_state: int = 8) -> None:
        super().__init__()
        self.dim = embed_dim
        self.grid = grid_size
        self.in_proj = nn.Linear(embed_dim, embed_dim)
        from .ssm3d import SelectiveSSM1D  # 3D-only module, not shipped in this release
        self.ssm = SelectiveSSM1D(embed_dim, d_state=d_state)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        b, L, d = x.shape
        g = self.grid
        V = L // (g * g)
        h = self.in_proj(x).view(b, V, g, g, d)
        out = 0.0
        for axis in (2, 3):  # rows, cols
            xp = h.movedim(axis, -2)               # [..., g, d]
            bsz = xp.shape[:-2]
            folded = xp.reshape(-1, g, d)
            # The length-g scan's autograd graph holds O(g) big per-step tensors;
            # checkpoint it (recompute in backward) to bound memory, as in the 3D SSM.
            if self.training and torch.is_grad_enabled():
                y = checkpoint(self.ssm, folded, use_reentrant=False)
            else:
                y = self.ssm(folded)
            y = y.reshape(*bsz, g, d)
            out = out + y.movedim(-2, axis)
        out = (out / 2.0).reshape(b, L, d)
        return self.out_proj(out)


class RadialSelfAttention2D(nn.Module):
    """Full softmax attention with a learnable per-head distance-decay bias
    (logits -= softplus(alpha_h) * spatial_distance) -- the 'energy decay' radial
    prior. Dense O(L^2) (fine at 2D scale); tests the decay prior's quality before
    we build the sub-quadratic sparse 3D version. Distance is by spatial position
    (field-agnostic), so spatially-aligned cross-field pairs are unpenalised."""

    def __init__(self, embed_dim: int, num_heads: int, grid_size: int) -> None:
        super().__init__()
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid = grid_size
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.log_alpha = nn.Parameter(torch.zeros(num_heads))  # decay strength per head
        g = grid_size
        c = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij"), dim=-1).reshape(g * g, 2).float()
        self.register_buffer("dist", torch.cdist(c, c), persistent=False)  # [HW, HW]
        self._bias = None  # cached [L, L] spatial-distance bias

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def _bias_full(self, L: int, device) -> torch.Tensor:
        if self._bias is None or self._bias.shape[0] != L or self._bias.device != device:
            hw = self.grid * self.grid
            sp = torch.arange(L, device=device) % hw
            self._bias = self.dist.to(device)[sp][:, sp]  # [L, L]
        return self._bias

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        b, L, d = x.shape
        H, hd = self.num_heads, self.head_dim
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        q, k = apply_2d_rope(q, k, spatial_indices=spatial_indices, grid_width=self.grid)
        # distance decay as an additive attention bias passed to SDPA, so the
        # memory-efficient backend streams it (O(L) memory) instead of us
        # materialising a [b,H,L,L] score matrix. [1,H,L,L] broadcasts over batch.
        # (KH has all fields valid, so no key-validity term is needed here.)
        alpha = F.softplus(self.log_alpha).view(1, H, 1, 1)
        bias = (-alpha * self._bias_full(L, x.device).view(1, 1, L, L)).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(b, L, d)
        return self.out_proj(out)


class QuadTreeSelfAttention2D(nn.Module):
    """2-level block-wise QuadTree attention (coarse-to-fine, adaptive top-K).

    Per field, the g x g grid is pooled into a (g/pool)^2 grid of coarse regions
    (pool x pool fine tokens each). For each region, coarse region-region scores
    (mean-pooled q.k) pick its `topk` most-relevant regions; the region's fine
    queries then attend (full softmax) to the fine tokens of just those regions.
    Block-level selection (shared by a region's queries) keeps the gather memory
    bounded. With topk == num_regions this reduces exactly to full within-field
    attention (asserted in the correctness test). Within-field, like windowed/ssm."""

    def __init__(self, embed_dim: int, num_heads: int, grid_size: int, pool: int = 4, topk: int = 8) -> None:
        super().__init__()
        if (embed_dim // num_heads) % 4 != 0:
            raise ValueError("Per-head dim must be divisible by 4 for 2D RoPE")
        if grid_size % pool != 0:
            raise ValueError(f"pool {pool} must divide grid_size {grid_size}")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.grid = grid_size
        self.pool = pool
        self.cg = grid_size // pool
        self.C = self.cg * self.cg          # coarse regions per field
        self.R = pool * pool                # fine tokens per region
        self.topk = min(topk, self.C)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _to_regions(self, t: torch.Tensor, b: int, V: int) -> torch.Tensor:
        # [b, H, L, hd] -> [b*H*V, C, R, hd]
        H, hd, p, cg = self.num_heads, self.head_dim, self.pool, self.cg
        t = t.view(b, H, V, cg, p, cg, p, hd).permute(0, 1, 2, 3, 5, 4, 6, 7)
        return t.reshape(b * H * V, self.C, self.R, hd)

    def _core(self, qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor, b: int, V: int) -> torch.Tensor:
        # qh/kh/vh: [b,H,L,hd] (RoPE'd) -> attention output [b, L, d] (pre out_proj)
        H, hd, g, p, cg = self.num_heads, self.head_dim, self.grid, self.pool, self.cg
        C, R, K = self.C, self.R, self.topk
        L = V * g * g
        qr = self._to_regions(qh, b, V)      # [N, C, R, hd], N = b*H*V
        kr = self._to_regions(kh, b, V)
        vr = self._to_regions(vh, b, V)
        N = qr.shape[0]
        # coarse region descriptors -> region-region scores -> top-K regions
        qc, kc = qr.mean(dim=2), kr.mean(dim=2)                       # [N, C, hd]
        coarse = torch.matmul(qc, kc.transpose(-2, -1)) * (hd ** -0.5)  # [N, C, C]
        idx = coarse.topk(K, dim=-1).indices                          # [N, C, K]
        ar = torch.arange(N, device=qh.device)[:, None, None]
        gk = kr[ar, idx].reshape(N, C, K * R, hd)                     # gather fine k/v of top-K regions
        gv = vr[ar, idx].reshape(N, C, K * R, hd)
        scores = torch.matmul(qr, gk.transpose(-2, -1)) * (hd ** -0.5)  # [N, C, R, K*R]
        attn = torch.softmax(scores.float(), dim=-1).to(vr.dtype)
        out = torch.matmul(attn, gv)                                  # [N, C, R, hd]
        out = out.view(b, H, V, cg, cg, p, p, hd).permute(0, 1, 2, 3, 5, 4, 6, 7).reshape(b, H, L, hd)
        return out.transpose(1, 2).reshape(b, L, H * hd)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, spatial_indices: torch.Tensor) -> torch.Tensor:
        b, L, d = x.shape
        H, hd, g = self.num_heads, self.head_dim, self.grid
        V = L // (g * g)
        qh = self.q_proj(x).view(b, L, H, hd).transpose(1, 2)
        kh = self.k_proj(x).view(b, L, H, hd).transpose(1, 2)
        vh = self.v_proj(x).view(b, L, H, hd).transpose(1, 2)
        qh, kh = apply_2d_rope(qh, kh, spatial_indices=spatial_indices, grid_width=g)
        # The manual gather+softmax retains more than flash SDPA; checkpoint the
        # core so the encoder's stacked blocks don't accumulate it (recompute in
        # backward). top-K is deterministic, so recomputation is consistent.
        if self.training and torch.is_grad_enabled():
            out = checkpoint(self._core, qh, kh, vh, b, V, use_reentrant=False)
        else:
            out = self._core(qh, kh, vh, b, V)
        return self.out_proj(out)


def make_self_attention2d(mode: str, embed_dim: int, num_heads: int, grid_size: int) -> nn.Module:
    """Factory for the non-full attention modes (full stays inline in the block)."""
    mode = str(mode).lower().strip()
    if mode == "linear":
        return LinearSelfAttention2D(embed_dim, num_heads, grid_size)
    if mode == "windowed":
        return WindowedSelfAttention2D(embed_dim, num_heads, grid_size)
    if mode == "ssm":
        return SSM2DSelfAttention(embed_dim, num_heads, grid_size)
    if mode == "radial":
        return RadialSelfAttention2D(embed_dim, num_heads, grid_size)
    if mode == "quadtree":
        return QuadTreeSelfAttention2D(embed_dim, num_heads, grid_size)
    raise ValueError(f"unknown attention_mode '{mode}' (expected full|linear|windowed|ssm|radial|quadtree)")
