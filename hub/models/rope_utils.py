from __future__ import annotations

import math

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def axis_even_split(head_dim: int, num_axes: int = 3) -> list[int]:
    """Split head_dim into num_axes even-sized parts (each >= 2) for axis-wise RoPE."""
    if head_dim % 2 != 0:
        raise ValueError("Per-head dimension must be even for RoPE")
    base = (head_dim // num_axes) - ((head_dim // num_axes) % 2)
    dims = [base] * (num_axes - 1)
    dims.append(head_dim - base * (num_axes - 1))
    if any(d < 2 or d % 2 != 0 for d in dims):
        raise ValueError(f"Cannot split head_dim={head_dim} into {num_axes} even parts")
    return dims


def build_3d_rope_cache(
    coords: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute interleaved-pair RoPE cos/sin for 3D coordinates.

    coords: [L, 3] integer/float grid coordinates.
    Returns (cos, sin), each [L, head_dim], float32. The head dim is split into
    three even sub-blocks, one per axis, matching the 2D convention of
    apply_2d_rope (rotate_half over adjacent even/odd pairs).
    """
    dims = axis_even_split(int(head_dim), 3)
    coords = coords.float()
    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for axis, d_axis in enumerate(dims):
        half = d_axis // 2
        inv_freq = torch.exp(
            -math.log(base) * torch.arange(0, half, dtype=torch.float32) / max(1, half)
        )
        phase = coords[:, axis : axis + 1] * inv_freq.unsqueeze(0)
        cos_parts.append(torch.repeat_interleave(torch.cos(phase), repeats=2, dim=-1))
        sin_parts.append(torch.repeat_interleave(torch.sin(phase), repeats=2, dim=-1))
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply precomputed RoPE to one tensor x of shape [B', L, H, Dh].

    cos/sin: [L, Dh] (shared across the folded batch) or [nW, L, Dh] for
    windowed attention where B' = B_outer * nW (windows fastest-varying).
    Rotation is computed in float32 and cast back to the input dtype. Used on
    its own for halo cross-attention, where q (core) and k (haloed) have
    different lengths and so need separate caches.
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    n_win, seq_len, head_dim = cos.shape
    b_fold, x_len, n_heads, x_head_dim = x.shape
    if x_len != seq_len or x_head_dim != head_dim or b_fold % n_win != 0:
        raise ValueError(
            f"RoPE cache mismatch: x={tuple(x.shape)} cache={tuple(cos.shape)}"
        )
    c = cos.view(1, n_win, seq_len, 1, head_dim)
    s = sin.view(1, n_win, seq_len, 1, head_dim)
    xv = x.view(-1, n_win, seq_len, n_heads, head_dim).float()
    out = xv * c + rotate_half(xv) * s
    return out.view(b_fold, seq_len, n_heads, head_dim).to(x.dtype)


def apply_rope_cached(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same precomputed RoPE to q and k (identical L). See
    apply_rope_single for the cache layout."""
    return apply_rope_single(q, cos, sin), apply_rope_single(k, cos, sin)


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    spatial_indices: torch.Tensor,
    grid_width: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [B, H, L, Dh]
    head_dim = q.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError("Per-head dimension must be divisible by 4 for 2D RoPE")

    valid = spatial_indices >= 0
    clamped = spatial_indices.clamp(min=0)
    x_pos = (clamped % grid_width).to(q.dtype)
    y_pos = torch.div(clamped, grid_width, rounding_mode="floor").to(q.dtype)

    half_dim = head_dim // 2
    quarter_dim = half_dim // 2
    inv_freq = torch.exp(
        -math.log(base)
        * torch.arange(0, quarter_dim, device=q.device, dtype=q.dtype)
        / max(1, quarter_dim)
    )

    x_phase = x_pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    y_phase = y_pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)

    x_cos = torch.repeat_interleave(torch.cos(x_phase), repeats=2, dim=-1).unsqueeze(1)
    x_sin = torch.repeat_interleave(torch.sin(x_phase), repeats=2, dim=-1).unsqueeze(1)
    y_cos = torch.repeat_interleave(torch.cos(y_phase), repeats=2, dim=-1).unsqueeze(1)
    y_sin = torch.repeat_interleave(torch.sin(y_phase), repeats=2, dim=-1).unsqueeze(1)

    qx, qy = q[..., :half_dim], q[..., half_dim:]
    kx, ky = k[..., :half_dim], k[..., half_dim:]

    qx_rot = (qx * x_cos) + (rotate_half(qx) * x_sin)
    kx_rot = (kx * x_cos) + (rotate_half(kx) * x_sin)
    qy_rot = (qy * y_cos) + (rotate_half(qy) * y_sin)
    ky_rot = (ky * y_cos) + (rotate_half(ky) * y_sin)

    q_rot = torch.cat([qx_rot, qy_rot], dim=-1)
    k_rot = torch.cat([kx_rot, ky_rot], dim=-1)

    valid_mask = valid.unsqueeze(1).unsqueeze(-1)
    q_out = torch.where(valid_mask, q_rot, q)
    k_out = torch.where(valid_mask, k_rot, k)
    return q_out, k_out
