import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd

from typing import Any, Sequence, Dict, Optional, Tuple, Literal
import numpy as np
import math
from einops import rearrange, repeat
from functools import partial

from tokenizer.model.building_blocks.blocks.block_utils import default_init, reshape_jax_torch

Tensor = torch.Tensor


class CombineResidualWithSkip(nn.Module):
    """Combine residual and skip connections.

    Attributes:
      project_skip: Whether to add a linear projection layer to the skip
        connections. Mandatory if the number of channels are different between
        skip and residual values.
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_dim: int = None,
        project_skip: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Any | None = None,
    ):
        super(CombineResidualWithSkip, self).__init__()

        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_dim = kernel_dim
        self.project_skip = project_skip
        self.dtype = dtype
        self.device = device

        if residual_channels != skip_channels and not project_skip:
            raise ValueError(
                f"Residual tensor has {residual_channels}, Skip tensor has {skip_channels}. "
                f"Set project_skip to True to resolve this mismatch."
            )

        if self.residual_channels and self.skip_channels and self.project_skip:
            self.skip_projection = nn.Linear(
                skip_channels, residual_channels, device=self.device, dtype=self.dtype
            )
            torch.nn.init.kaiming_uniform_(self.skip_projection.weight, a=np.sqrt(5))
            torch.nn.init.zeros_(self.skip_projection.bias)
        else:
            self.skip_projection = None

    def forward(self, residual: Tensor, skip: Tensor) -> Tensor:
        # residual, skip (bs, c, w, h, d)
        if self.project_skip:
            skip = self.skip_projection(skip.permute(0, 3, 2, 1)).permute(0,3,2,1)  # (bs, w, h, c) -> (bs, c, w, h)
            # skip = reshape_jax_torch(
            #     self.skip_projection(reshape_jax_torch(skip, self.kernel_dim)),
            #     self.kernel_dim,
            # )

        return (skip + residual) / np.sqrt(2)

class MultiHeadDotProductAttention(nn.Module):
    """Mulit Head Dot Product Attention with querry and key normalization"""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        normalize_qk: bool = False,
        dropout: float = 0.0,
        device: Any | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(MultiHeadDotProductAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        if emb_dim % num_heads != 0:
            raise ValueError(
                "Embedding Dimension must be divisible through the number of heads"
            )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
            device=self.device,
            dtype=self.dtype,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.multihead_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.multihead_attention.out_proj.weight)

    def forward(
        self, query: Tensor, key: Tensor = None, value: Tensor = None
    ) -> Tensor:
        """Required shape for multihead attention is:

        2D case: (bs, width*height, emb_dim)
        3D case: (bs, length, emb_dim)
        where the length is just the height, width or depth. Used for axial self attention
        """

        if key is None and value is None:
            key = value = query

        elif key is None:
            if value is not None:
                raise ValueError("value can not be not None if key is None")
            key = query

        if value is None:
            value = key

        if self.normalize_qk:
            # L2 normalization across the feature dimension
            query = F.normalize(query, p=2, dim=-1)
            key = F.normalize(key, p=2, dim=-1)

        out, _ = self.multihead_attention(query, key, value)

        return out

class AttentionBlock(nn.Module):
    """Attention block."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1,
        normalize_qk: bool = False,
    ):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk

        self.norm = nn.GroupNorm(
            min(max(self.in_channels // 4, 1), 32),
            self.in_channels,
        )

        self.multihead_attention = MultiHeadDotProductAttention(
            emb_dim=self.in_channels,
            num_heads=self.num_heads,
            dropout=0.1,
        )

        self.res_layer = CombineResidualWithSkip(
            residual_channels=in_channels,
            skip_channels=in_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input x -> (bs, widht*height, c)
        h = x.clone()
        # GroupNorm requires x -> (bs, c, widht*height)
        h = self.norm(h.permute(0, 2, 1))
        h = h.permute(0, 2, 1)  # (bs, width*height, c)
        h = self.multihead_attention(h, h, h)  # Selfattention
        h = self.res_layer(residual=h, skip=x)

        return h


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale


# class LinearAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         heads = 4,
#         dim_head = 32,
#         num_mem_kv = 4
#     ):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads

#         self.norm = RMSNorm(dim)

#         self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Conv2d(hidden_dim, dim, 1),
#             RMSNorm(dim)
#         )

#     def forward(self, x):
#         b, c, h, w = x.shape

#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim = 1)
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

#         mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
#         k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

#         q = q.softmax(dim = -2)
#         k = k.softmax(dim = -1)

#         q = q * self.scale

#         context = torch.einsum('b h d n, b h e n -> b h d e', k, v)


#         out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
#         out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
#         return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        hidden_dim = dim_head * heads
        
        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        
        # Get QKV in one go
        qkv = self.to_qkv(x)  # b, hidden_dim*3, h, w
        
        # Reshape more efficiently
        qkv = qkv.view(b, 3, self.heads, self.dim_head, h * w)
        q, k, v = qkv.unbind(dim=1)  # Each: b, heads, dim_head, h*w
        
        # Add memory kv more efficiently
        mk, mv = self.mem_kv.unbind(dim=0)  # Each: heads, dim_head, num_mem_kv
        mk = mk.unsqueeze(0).expand(b, -1, -1, -1)  # b, heads, dim_head, num_mem_kv
        mv = mv.unsqueeze(0).expand(b, -1, -1, -1)
        
        k = torch.cat([mk, k], dim=-1)  # b, heads, dim_head, num_mem_kv + h*w
        v = torch.cat([mv, v], dim=-1)
        
        # Apply softmax and scaling
        q = F.softmax(q, dim=-2) * self.scale
        k = F.softmax(k, dim=-1)
        
        # Use bmm instead of einsum for better performance
        # context = k @ v.transpose(-2, -1)  # b*heads, dim_head, dim_head
        # out = context @ q  # b*heads, dim_head, h*w
        
        # Reshape for batch matrix multiplication
        b_heads = b * self.heads
        q_flat = q.view(b_heads, self.dim_head, h * w)
        k_flat = k.view(b_heads, self.dim_head, -1)
        v_flat = v.view(b_heads, self.dim_head, -1)
        
        # More efficient attention computation
        context = torch.bmm(k_flat, v_flat.transpose(-2, -1))  # b*heads, dim_head, dim_head
        out = torch.bmm(context, q_flat)  # b*heads, dim_head, h*w
        
        # Reshape back
        out = out.view(b, self.heads * self.dim_head, h, w)
        
        return self.to_out(out)


# ----------------------------------------------------------------
@torch.no_grad()
def _positive_stable(alpha: float, size, device, dtype) -> torch.Tensor:
    """
    One‑sided positive‑stable random variable with index ``alpha`` in ``(0,1]``.
    For ``alpha=1`` this degenerates to 1.  Implements the Kanter/CMS sampler.
    """
    if alpha >= 1.0 - 1e-8:
        return torch.ones(size, device=device, dtype=dtype)
    U = torch.rand(size, device=device, dtype=dtype) * (math.pi / 2.0)
    E = -torch.log(torch.rand(size, device=device, dtype=dtype).clamp_min(1e-12))
    s = torch.sin(alpha * U) / (torch.cos(U)).pow(1.0 / alpha)
    c = torch.cos((1.0 - alpha) * U) / E
    X = s * c.pow((1.0 - alpha) / alpha)
    return X.clamp_min(1e-38)


@torch.no_grad()
def _sample_rff_matrix_alpha_stable(
    h: int, d: int, D: int, alpha: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Sample an isotropic SαS random matrix W in R^{h×d×D} via the
    sub‑Gaussian mixture representation:

        W = sqrt(P) * Z,    Z ∼ N(0, I_d),    P ∼ PS(alpha/2).

    For alpha = 2 this reduces to a Gaussian distribution.  Returns
    W with shape ``[h, d, D]`` where each column is i.i.d. isotropic SαS.
    """
    if alpha >= 2.0 - 1e-8:
        return torch.randn(h, d, D, device=device, dtype=dtype)
    P = _positive_stable(alpha / 2.0, (h, 1, D), device=device, dtype=dtype)
    Z = torch.randn(h, d, D, device=device, dtype=dtype)
    return Z * torch.sqrt(P)


@torch.no_grad()
def fast_qr(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Perform a reduced QR decomposition returning orthonormal columns.
    Uses ``torch.linalg.qr`` if available; falls back to a simple
    Gram–Schmidt process otherwise.
    """
    if hasattr(torch.linalg, "qr"):
        Q, _ = torch.linalg.qr(X, mode='reduced')
        return Q
    # Fallback Gram–Schmidt
    Qcols = []
    for j in range(X.shape[1]):
        v = X[:, j]
        for q in Qcols:
            v = v - (q @ v) * q
        n2 = (v * v).sum()
        v = v * torch.rsqrt(n2 + eps)
        Qcols.append(v)
    return torch.stack(Qcols, 1)


# Types for readability
Backend = Literal["rff", "nystrom"]
KernelType = Literal["gaussian", "frac", "drift", "fracdrift"]
LandmarkMode = Literal["learned", "kmeans_once"]


class SemigroupAttention(nn.Module):
    """
    Streaming O(N·D) semigroup attention with positive features and optional
    Coifman–Lafon α‑normalisation on both sides, plus an optional symmetric
    realisation (D^{-1/2} K D^{-1/2}) for exact non‑expansiveness in the
    stationary metric.

    This class mirrors the original reference implementation but adds
    several performance improvements.  See the module docstring for a
    high‑level summary of the optimisation strategy.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        *,
        backend: Backend = "rff",
        rff_dim: int = 128,
        chunk_size: int = 4096,
        query_block: int = 1024,
        eig_dim: int = 16,
        eig_update_freq: int = 2000,
        use_low_rank: bool = False,
        learn_sigma: bool = False,
        base_sigma: float = 1.0,
        eps: float = 1e-6,
        kernel_type: KernelType = "fracdrift",
        drift_init: float = 0.3,
        frac_alpha: float = 2.0,
        alpha_cl: float = 0.0,
        symmetric_realization: bool = False,
        landmark_mode: LandmarkMode = "learned",
        kmeans_sample: int = 20000,
        kmeans_iters: int = 5,
        causal: bool = False,
        no_self: bool = False,
        causal_block: int = 128,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        assert 0.0 <= alpha_cl <= 1.0
        if backend == "nystrom":
            assert kernel_type in {"gaussian", "drift"}, (
                "Nyström backend only supports 'gaussian' or 'drift' kernels."
            )

        self.h = num_heads
        self.d = dim // num_heads
        self.D = int(rff_dim)
        self.chunk = int(chunk_size)
        self.query_block = int(query_block)
        self.causal_block = int(causal_block)
        self.eps = float(eps)
        self.backend = backend
        self.kernel_type = kernel_type
        self.alpha_cl = float(alpha_cl)
        self.symmetric_realization = bool(symmetric_realization)
        self.landmark_mode = landmark_mode
        self.kmeans_sample = int(kmeans_sample)
        self.kmeans_iters = int(kmeans_iters)
        self.causal = bool(causal)
        self.no_self = bool(no_self)

        # input projection (Q = K by design) and output projection
        self.qv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # learnable bandwidth σ; stored in log space for stability.  We
        # register the buffer with persistent=False so that Lightning
        # excludes it from state dicts when sigma is not learnable.
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(base_sigma)))
        else:
            self.register_buffer(
                "log_sigma",
                torch.log(torch.tensor(base_sigma)),
                persistent=False,
            )

        # drift parameter per head (tilting)
        if kernel_type in {"drift", "fracdrift"}:
            self.drift = nn.Parameter(torch.full((self.h, self.d), float(drift_init)))
        else:
            self.register_buffer("drift", None, persistent=False)

        # backend parameters
        if self.backend == "rff":
            self.frac_alpha = float(frac_alpha)
            # Sample the random feature matrix on the target device up front.
            # If CUDA is available the matrix is allocated directly on the
            # GPU, which avoids a costly CPU→GPU copy on the first forward
            # pass.  Dtype is initially FP32; during the first call to
            # _phi_pos_rff it is promoted to match the incoming tensor.
            rff_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            W = _sample_rff_matrix_alpha_stable(
                self.h,
                self.d,
                self.D,
                self.frac_alpha,
                device=rff_device,
                dtype=torch.float32,
            )
            self.register_buffer("W_rff", W, persistent=False)
            b = torch.rand(self.h, self.D, device=rff_device, dtype=torch.float32) * 2 * math.pi
            self.register_buffer("b_rff", b, persistent=False)
            self.feat_scale = 1.0  # IMPORTANT: no dim^{-1/2} scaling
            self.register_buffer("landmarks", None, persistent=False)
            self._landmarks_inited = True
        else:
            # Nyström landmarks are either learned or lazily initialised.
            if landmark_mode == "learned":
                self.landmarks = nn.Parameter(torch.randn(self.h, self.D, self.d) * 0.02)
                self._landmarks_inited = True
            else:
                self.register_buffer("landmarks", torch.empty(self.h, self.D, self.d), persistent=False)
                self._landmarks_inited = False
            self.register_buffer("W_rff", None, persistent=False)
            self.register_buffer("b_rff", None, persistent=False)
            self.feat_scale = 1.0

        # optional tiny low‑rank projection of the outputs
        self.eig_dim = int(eig_dim)
        self.eig_update_freq = int(eig_update_freq)
        self.use_low_rank = bool(use_low_rank and self.eig_dim > 0)
        self.register_buffer("eig_vecs", None, persistent=False)
        self.update_counter = 0

        # cache for current time (optional)
        self._current_time: Optional[float] = None

        # internal flag to lazily promote RFF parameters to match input
        # dtype/device.  This avoids repeated ``.to()`` calls inside
        # _phi_pos_rff.
        self._rff_initialized = False

    # ---------------------- user knobs ----------------------

    @torch.no_grad()
    def set_time(self, t: float) -> None:
        """
        Set semigroup time via ``t = sigma^{-alpha}  =>  sigma = t^{-1/alpha}``.
        Modifies the log_sigma buffer in place.  See the original
        implementation for details.
        """
        alpha = getattr(self, "frac_alpha", 2.0)
        sigma = max(t, 1e-12) ** (-1.0 / float(alpha))
        # When sigma is learnable the parameter is updated; when sigma is
        # a buffer the copy_ call suffices.  Note: converting to
        # ``log_sigma.dtype`` avoids dtype mismatches.
        self.log_sigma.copy_(
            torch.tensor(math.log(sigma), device=self.log_sigma.device, dtype=self.log_sigma.dtype)
        )
        self._current_time = nn.Parameter(t*torch.ones(1), requires_grad=True)

    # ---------------------- landmarks (Nyström) ----------------------

    @torch.no_grad()
    def _init_landmarks_once(self, q: torch.Tensor) -> None:
        if self._landmarks_inited or self.backend != "nystrom":
            return
        B, H, N, d = q.shape
        m = self.D
        new_centers = []
        S = min(self.kmeans_sample, B * N)
        for hh in range(H):
            X = q[:, hh].reshape(B * N, d)
            if X.shape[0] == 0:
                new_centers.append(torch.zeros(m, d, device=X.device, dtype=X.dtype))
                continue
            idx = torch.randint(0, X.shape[0], (S,), device=X.device)
            Xs = X[idx].contiguous()
            # kmeans++ init
            pick = torch.randint(0, Xs.shape[0], (1,), device=X.device)
            centers = [Xs[pick]]
            for _ in range(1, m):
                C = torch.cat(centers, dim=0)
                dist2 = (Xs ** 2).sum(-1, keepdim=True) + (C ** 2).sum(-1)[None, :] - 2 * (Xs @ C.T)
                min_d2 = dist2.min(dim=1).values
                probs = (min_d2 / (min_d2.sum() + 1e-12)).clamp_min(1e-12)
                next_idx = torch.multinomial(probs, 1)
                centers.append(Xs[next_idx])
            C = torch.cat(centers, dim=0)
            for _ in range(max(0, self.kmeans_iters)):
                dist2 = (Xs ** 2).sum(-1, keepdim=True) + (C ** 2).sum(-1)[None, :] - 2 * (Xs @ C.T)
                assign = dist2.argmin(dim=1)
                for j in range(m):
                    mask = assign == j
                    if mask.any():
                        C[j] = Xs[mask].mean(dim=0)
            new_centers.append(C)
        C = torch.stack(new_centers, dim=0)  # [h,m,d]
        if isinstance(getattr(self, "landmarks"), nn.Parameter):
            self.landmarks.detach().copy_(C)
        else:
            self.landmarks = C
        self._landmarks_inited = True

    # ---------------------- positive feature maps ----------------------
    def _phi_pos_rff(self, x: torch.Tensor, sigma: torch.Tensor, is_key: bool) -> torch.Tensor:
        # lazily promote RFF params to match the input dtype/device once
        W = self.W_rff
        b = self.b_rff
        if W.device != x.device or W.dtype != x.dtype:
            # create temporaries; don't assign back to self.*
            W = W.to(device=x.device, dtype=x.dtype)
            b = b.to(device=x.device, dtype=x.dtype)

        phase = (x / sigma[..., None]) @ W + b[None, :, None, :]  # [B,h,T,D]
        feat = self.feat_scale * torch.cos(phase)
        feat = F.elu(feat, alpha=1.0) + 1.0
        if self.kernel_type in {"drift", "fracdrift"} and self.drift is not None:
            drift_dot = (x * self.drift[None, :, None, :]).sum(-1, keepdim=True).clamp(-10.0, 10.0)
            feat = feat * torch.exp(drift_dot if is_key else -drift_dot)
        return feat

    def _phi_pos_nystrom(self, x: torch.Tensor, sigma: torch.Tensor, is_key: bool) -> torch.Tensor:
        L = self.landmarks.to(device=x.device, dtype=x.dtype)  # [h,m,d]
        x2 = (x * x).sum(dim=-1, keepdim=True)
        l2 = (L * L).sum(dim=-1)[None, :, None, :]
        xL = torch.einsum('B h T d, h m d -> B h T m', x, L)
        sigma2 = (sigma * sigma).view(1, 1, 1, 1)
        feat = torch.exp(-0.5 * (x2 + l2 - 2.0 * xL).clamp_min(0.0) / (sigma2 + 1e-12))
        if self.kernel_type in {"drift"} and self.drift is not None:
            drift_dot = (x * self.drift[None, :, None, :]).sum(-1, keepdim=True).clamp(-10.0, 10.0)
            feat = feat * torch.exp(drift_dot if is_key else -drift_dot)
        return feat

    def _phi_pos(self, x: torch.Tensor, sigma: torch.Tensor, is_key: bool) -> torch.Tensor:
        return self._phi_pos_rff(x, sigma, is_key) if self.backend == "rff" else self._phi_pos_nystrom(x, sigma, is_key)

    # --------------------------- forward ---------------------------
    def forward(self, x: torch.Tensor, dropout_p: float = 0.0, causal: Optional[bool] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Apply the semigroup attention to the input.  The semantics of
        this method are identical to the reference implementation.  See
        the original code for detailed comments on each section.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, N, dim]`` or pre‑projected
            ``[B, N, 2, H, d]`` containing queries/keys and values.
        dropout_p : float, optional
            Dropout probability (unused; kept for API compatibility).
        causal : bool, optional
            If provided, overrides the ``causal`` attribute for this
            forward pass.
        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[B, N, dim]``.
        """
        H, D, d = self.h, self.D, self.d
        causal_flag = self.causal if causal is None else bool(causal)

        # accept either [B,N,dim] or preprojected [B,N,2,H,d]
        if x.dim() == 5:
            B, N, two, H_in, d_in = x.shape
            assert two == 2 and H_in == H and d_in == d
            q, v = x.unbind(dim=2)  # [B,N,H,d]
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
        elif x.dim() == 3:
            B, N, Dm = x.shape
            assert Dm == H * d
            qv = self.qv_proj(x)
            q_raw, v_raw = qv.split([Dm, Dm], dim=-1)
            q = q_raw.view(B, N, H, d).transpose(1, 2)
            v = v_raw.view(B, N, H, d).transpose(1, 2)
        else:
            raise ValueError("SemigroupAttention expects [B,N,dim] or [B,N,2,H,d].")

        if self.backend == "nystrom" and not getattr(self, "_landmarks_inited"):
            with torch.no_grad():
                self._init_landmarks_once(q.to(torch.float32))

        sigma = self.log_sigma.exp().to(q.dtype)
        out = q.new_empty(B, H, N, d)

        # exponents for CL normalisation and symmetric realisation
        exp_k = self.alpha_cl + (0.5 if self.symmetric_realization else 0.0)
        exp_q = self.alpha_cl + (0.5 if self.symmetric_realization else 0.0)

        if causal_flag:
            # causal scan (block‑wise)
            do_weight = (exp_k > 0.0) or (exp_q > 0.0)
            # use float32 accumulators for numerical stability
            R_prev_unw = torch.zeros(B, H, D, dtype=torch.float32, device=q.device)
            M_prev_unw = torch.zeros(B, H, D, d, dtype=torch.float32, device=q.device)
            R_prev_w = torch.zeros_like(R_prev_unw) if do_weight else R_prev_unw
            M_prev_w = torch.zeros_like(M_prev_unw) if do_weight else M_prev_unw

            for i0 in range(0, N, self.chunk):
                i1 = min(N, i0 + self.chunk)
                q_chunk = q[:, :, i0:i1]
                v_chunk32 = v[:, :, i0:i1].to(torch.float32)

                # per‑chunk accumulators
                R_acc_unw = torch.zeros_like(R_prev_unw)
                M_acc_unw = torch.zeros_like(M_prev_unw)
                R_acc_w = torch.zeros_like(R_prev_w)
                M_acc_w = torch.zeros_like(M_prev_w)

                # process the chunk in blocks of causal_block tokens
                for j0 in range(0, i1 - i0, self.causal_block):
                    j1 = min(i1 - i0, j0 + self.causal_block)
                    Tloc = j1 - j0

                    q_blk = q_chunk[:, :, j0:j1]  # [B,h,Tb,d]
                    fq_unw = self._phi_pos(q_blk, sigma, is_key=False)  # [B,h,Tb,D]
                    fk_unw = self._phi_pos(q_blk, sigma, is_key=True)  # [B,h,Tb,D]
                    # prefix sums for "past only" on unweighted keys
                    prefix_unw = torch.cumsum(fk_unw.to(torch.float32), dim=2)  # [B,h,Tb,D]
                    prefix_unw_excl = torch.roll(prefix_unw, 1, dims=2)
                    prefix_unw_excl[:, :, 0] = 0
                    # degrees for query/key weighting computed AGAINST UNWEIGHTED degrees
                    deg_q = (fq_unw.to(torch.float32) * (R_prev_unw.unsqueeze(2) + prefix_unw_excl)).sum(-1).clamp_min(self.eps)
                    deg_k = (fk_unw.to(torch.float32) * (R_prev_unw.unsqueeze(2) + prefix_unw_excl)).sum(-1).clamp_min(self.eps)
                    if exp_q > 0:
                        wq = deg_q.pow(-exp_q).unsqueeze(-1)
                    else:
                        wq = 1.0
                    if exp_k > 0:
                        wk = deg_k.pow(-exp_k).unsqueeze(-1)
                    else:
                        wk = 1.0
                    fq = fq_unw * wq
                    fk = fk_unw * wk
                    # weighted prefix sums for denominators
                    prefix_w = torch.cumsum(fk.to(torch.float32), dim=2)
                    prefix_w_excl = torch.roll(prefix_w, 1, dims=2)
                    prefix_w_excl[:, :, 0] = 0
                    Rsum_w = (R_prev_w + R_acc_w).unsqueeze(2) + prefix_w_excl  # [B,h,Tb,D]
                    denom = (fq.to(Rsum_w.dtype) * Rsum_w).sum(-1).clamp_min(self.eps)
                    # numerator: contributions from previous blocks
                    num_prev = torch.einsum('B h T D, B h D d -> B h T d', fq.to(torch.float32), M_prev_w)
                    # contribution from within the current block
                    if self.symmetric_realization and (exp_k > 0):
                        v_blk32 = (v_chunk32[:, :, j0:j1] * wk.squeeze(-1)).to(torch.float32)  # [B,h,Tb,d]
                    else:
                        v_blk32 = v_chunk32[:, :, j0:j1]
                    delta_M = torch.einsum('B h T D, B h T d -> B h T D d', fk.to(torch.float32), v_blk32)
                    cumsum_M = torch.cumsum(delta_M, dim=2)
                    num_in_blk = torch.einsum('B h T D, B h T D d -> B h T d', fq.to(torch.float32), cumsum_M)
                    num_blk = (num_prev + num_in_blk).to(out.dtype)
                    out[:, :, i0 + j0:i0 + j1] = (num_blk / denom.unsqueeze(-1)).to(out.dtype)
                    R_acc_unw += fk_unw.sum(dim=2, dtype=torch.float32)
                    M_acc_unw += torch.einsum('B h T D, B h T d -> B h D d', fk_unw.to(torch.float32), v_chunk32[:, :, j0:j1])
                    R_acc_w += fk.sum(dim=2, dtype=torch.float32)
                    if self.symmetric_realization and (exp_k > 0):
                        V_blk = v_blk32  # already weighted by wk
                    else:
                        V_blk = v_chunk32[:, :, j0:j1]
                    M_acc_w += torch.einsum('B h T D, B h T d -> B h D d', fk.to(torch.float32), V_blk.to(torch.float32))
                    # delete temporaries early to release memory (optional but kept for clarity)
                    del q_blk, fq_unw, fk_unw, fq, fk, prefix_unw, prefix_unw_excl
                    del prefix_w, prefix_w_excl, Rsum_w, denom
                    del num_prev, v_blk32, delta_M, cumsum_M, num_in_blk, num_blk, V_blk
                R_prev_unw += R_acc_unw
                M_prev_unw += M_acc_unw
                R_prev_w += R_acc_w
                M_prev_w += M_acc_w
                del q_chunk, v_chunk32, R_acc_unw, M_acc_unw, R_acc_w, M_acc_w

        else:
            # non‑causal: two scans (key) + micro‑block (query)
            R1 = torch.zeros(B, H, D, dtype=torch.float32, device=q.device)
            M1 = torch.zeros(B, H, D, d, dtype=torch.float32, device=q.device)
            for i in range(0, N, self.chunk):
                qk = q[:, :, i:i + self.chunk]
                vk = v[:, :, i:i + self.chunk]
                fk_unw = self._phi_pos(qk, sigma, is_key=True)
                R1 += fk_unw.sum(dim=2, dtype=torch.float32)
                Bh, Tk = B * H, fk_unw.shape[2]
                F = fk_unw.reshape(Bh, Tk, D).transpose(1, 2).to(torch.float32)
                V = vk.reshape(Bh, Tk, d).to(torch.float32)
                M1 += (F @ V).reshape(B, H, D, d)
                del qk, vk, fk_unw, F, V
            need_weight = (exp_k > 0.0)
            if need_weight:
                R = torch.zeros_like(R1)
                M = torch.zeros_like(M1)
                for i in range(0, N, self.chunk):
                    qk = q[:, :, i:i + self.chunk]
                    vk = v[:, :, i:i + self.chunk]
                    fk_unw = self._phi_pos(qk, sigma, is_key=True)
                    fk32 = fk_unw.to(torch.float32)
                    deg = (fk32 * R1.unsqueeze(2)).sum(-1).clamp_min(self.eps)
                    w = deg.pow(-exp_k).unsqueeze(-1)
                    fk32 = fk32 * w
                    R += fk32.sum(2)
                    Bh, Tk = B * H, fk32.shape[2]
                    if self.symmetric_realization:
                        V = (vk * w.squeeze(-1)).reshape(Bh, Tk, d).to(torch.float32)
                    else:
                        V = vk.reshape(Bh, Tk, d).to(torch.float32)
                    F = fk32.reshape(Bh, Tk, D).transpose(1, 2)
                    M += (F @ V).reshape(B, H, D, d)
                    del qk, vk, fk_unw, fk32, deg, w, V, F
            else:
                R, M = R1, M1
            R_bh = R.to(q.dtype).reshape(B * H, D)
            M_bh = M.to(q.dtype).reshape(B * H, D, d)
            R1_bh = R1.reshape(B * H, D)
            for i in range(0, N, self.chunk):
                qq = q[:, :, i:i + self.chunk]
                vv = v[:, :, i:i + self.chunk]
                Tc = qq.shape[2]
                for j in range(0, Tc, self.query_block):
                    j1 = min(Tc, j + self.query_block)
                    tb = j1 - j
                    qq_blk = qq[:, :, j:j1]
                    vv_blk = vv[:, :, j:j1]
                    fq_unw = self._phi_pos(qq_blk, sigma, is_key=False)
                    F_blk = fq_unw.reshape(B * H, tb, D)
                    deg_q = (F_blk * R1_bh.unsqueeze(1)).sum(-1).clamp_min(self.eps)
                    if exp_q > 0:
                        wq = deg_q.pow(-exp_q).unsqueeze(-1)
                        F_blk = F_blk * wq
                    denom_all = F_blk.bmm(R_bh.unsqueeze(-1)).squeeze(-1)
                    num_all = F_blk.bmm(M_bh)
                    if self.no_self:
                        fk_same_unw = self._phi_pos(qq_blk, sigma, is_key=True)
                        FK_same = fk_same_unw.reshape(B * H, tb, D)
                        if need_weight:
                            deg_k = (FK_same * R1_bh.unsqueeze(1)).sum(-1).clamp_min(self.eps)
                            wk = deg_k.pow(-exp_k).unsqueeze(-1)
                            FK_same = FK_same * wk
                        if self.symmetric_realization and need_weight:
                            V_same = (vv_blk.reshape(B * H, tb, d) * wk.squeeze(-1))
                        else:
                            V_same = vv_blk.reshape(B * H, tb, d)
                        self_R = FK_same
                        self_M = self_R.unsqueeze(-1) * V_same.unsqueeze(-2)
                        self_denom = (F_blk * self_R).sum(-1)
                        self_num = (F_blk.unsqueeze(-1) * self_M).sum(-2)
                        denom = (denom_all - self_denom).clamp_min(self.eps)
                        num = num_all - self_num
                    else:
                        denom = denom_all
                        num = num_all
                    out[:, :, i + j:i + j1] = (num.reshape(B, H, tb, d) / denom.reshape(B, H, tb).unsqueeze(-1)).to(out.dtype)
                    del qq_blk, vv_blk, fq_unw, F_blk, denom_all, num_all
                del qq, vv

        if self.use_low_rank:                      # <- single gate
            if self.training and (self.update_counter % self.eig_update_freq == 0):
                with torch.no_grad():
                    self._update_eigen(out[0].to(torch.float32))
            self.update_counter += 1
            if self.eig_vecs is not None:
                Vb32  = self.eig_vecs.to(torch.float32)
                out32 = out.to(torch.float32)
                proj32 = torch.einsum("h N m, B h N d->B h m d", Vb32, out32)
                out    = torch.einsum("h N m, B h m d->B h N d", Vb32, proj32).to(out.dtype)
        out = out.transpose(1, 2).reshape(B, N, H * d)

        return self.out_proj(out)

    @torch.no_grad()
    def _update_eigen(self, out0: torch.Tensor) -> None:
        """
        Tiny per‑head orthonormal basis (heuristic).  Unchanged from the
        original implementation; retained here verbatim.
        """
        H, N, d = out0.shape
        m = int(self.eig_dim)
        if m <= 0:
            return
        m = min(m, N, d)
        new_basis = []
        for hh in range(H):
            X = out0[hh, :, :m]
            Q = fast_qr(X)
            if (self.eig_vecs is None) or (self.eig_vecs.shape[1] != N) or (self.eig_vecs.shape[2] != m):
                new_basis.append(Q)
            else:
                blend = 0.5
                Qb = blend * Q + (1.0 - blend) * self.eig_vecs[hh]
                Qm = fast_qr(Qb)
                new_basis.append(Qm)
        self.eig_vecs = torch.stack(new_basis, 0).to(out0.dtype)


def _pad_to_multiple(x, multiple, dim=-1):
    """Pad last two spatial dims so they are multiples of `multiple`. Returns padded tensor and pad amounts."""
    B, C, H, W = x.shape
    pad_h = (multiple - (H % multiple)) % multiple
    pad_w = (multiple - (W % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)



class WindowAttention2D(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift=False,
                 qkv_bias=True, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.shift = shift

        # projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        # relative position bias
        ws = window_size
        num_rel = (2 * ws - 1) * (2 * ws - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_rel, num_heads))
        coords = torch.arange(ws)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flat = coords.reshape(2, -1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += ws - 1
        rel_coords[:, :, 1] += ws - 1
        rel_coords[:, :, 0] *= 2 * ws - 1
        relative_index = rel_coords.sum(-1)
        self.register_buffer("relative_index", relative_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size
        shift_size = ws // 2 if self.shift else 0

        # permute to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # pad to multiple of window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        if self.shift:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))

        Hp, Wp = x.shape[1], x.shape[2]
        nH, nW = Hp // ws, Wp // ws

        # partition windows
        x_windows = x.view(B, nH, ws, nW, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        x_windows = self.norm(x_windows)

        # qkv projection
        qkv = self.qkv(x_windows)
        qkv = qkv.reshape(qkv.shape[0], ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (num_windows*B, heads, N, head_dim)

        # normalize q,k
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_index.view(-1)].view(ws*ws, ws*ws, -1)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = attn + bias
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(x_windows.shape[0], ws * ws, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # merge windows
        out = out.view(B, nH, nW, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)

        if self.shift:
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(1, 2))

        # remove padding
        out = out[:, :H, :W, :].contiguous()
        out = out.permute(0, 3, 1, 2)  # back to BCHW

        # residual connection
        return x.permute(0, 3, 1, 2)[:, :, :H, :W] + out

class AxialAttention2D(nn.Module):
    """
    Axial attention: first attend along height for each column, then along width for each row.
    Uses standard dot-product attention but applied to 1D sequences along axes.
    Very memory friendly for large aspect-ratio inputs.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_h = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.proj_h = nn.Conv1d(dim, dim, kernel_size=1)

        self.qkv_w = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.proj_w = nn.Conv1d(dim, dim, kernel_size=1, bias=True)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def _attend_1d(self, x1d, qkv_conv, proj_conv):
        """
        x1d: (B*other, C, L) where L is sequence length along axis
        returns: (B*other, C, L)
        """
        Bother, C, L = x1d.shape
        qkv = qkv_conv(x1d)  # Bother, 3*C, L
        qkv = qkv.reshape(Bother, 3, self.num_heads, self.head_dim, L)  # Bother,3,heads,hd,L
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, Bother, heads, L, hd
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: Bother, heads, L, hd

        # reshape to (Bother*heads, L, hd) for bmm
        q = q.reshape(Bother * self.num_heads, L, self.head_dim)
        k = k.reshape(Bother * self.num_heads, L, self.head_dim)
        v = v.reshape(Bother * self.num_heads, L, self.head_dim)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.bmm(attn, v)  # (Bother*heads, L, hd)
        out = out.reshape(Bother, self.num_heads, L, self.head_dim).permute(0,1,3,2).contiguous()  # Bother, heads, hd, L
        out = out.reshape(Bother, self.dim, L)
        out = proj_conv(out)
        out = self.proj_dropout(out)
        return out

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # attend along height: treat each column separately -> B*W sequences of length H
        x_h = x.permute(0, 3, 1, 2).contiguous()  # B, W, C, H
        x_h = x_h.view(B * W, C, H)
        out_h = self._attend_1d(x_h, self.qkv_h, self.proj_h)  # (B*W, C, H)
        out_h = out_h.view(B, W, C, H).permute(0, 2, 3, 1).contiguous()  # B, C, H, W

        # attend along width: treat each row separately -> B*H sequences length W
        x_w = out_h.permute(0, 2, 1, 3).contiguous()  # B, H, C, W
        x_w = x_w.view(B * H, C, W)
        out_w = self._attend_1d(x_w, self.qkv_w, self.proj_w)  # (B*H, C, W)
        out_w = out_w.view(B, H, C, W).permute(0, 2, 1, 3).contiguous()  # B, C, H, W

        return x + out_w  # residual


class GlobalPoolAttention(nn.Module):
    """
    Adds a small set of global tokens which attend to the entire feature map sparsely:
    * Global tokens query the full flattened map (cheap because #global << H*W).
    * Optionally we can let the map attend back to global tokens (cross-attend).
    Use this to provide global context cheaply.
    """

    def __init__(self, dim, num_global=8, num_heads=8, qkv_bias=True, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_global = num_global
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.global_tokens = nn.Parameter(torch.randn(1, num_global, dim))
        # proj to queries for global tokens; keys/vals come from flattened map
        self.q_global = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_map = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)
        self.out_map = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns x (same shape) optionally modified by global context
        """
        B, C, H, W = x.shape
        N = H * W
        # keys and values from map
        kv = self.kv_map(x).reshape(B, 2, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]  # each (B, heads, N, hd)
        # global queries
        g = self.global_tokens.expand(B, -1, -1)  # (B, G, C)
        q = self.q_global(g).reshape(B, self.num_global, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, heads, G, hd)
        # compute attn: for each head do (B*heads, G, hd) x (B*heads, hd, N)
        q = q.reshape(B * self.num_heads, self.num_global, self.head_dim)
        k = k.reshape(B * self.num_heads, N, self.head_dim)
        v = v.reshape(B * self.num_heads, N, self.head_dim)

        attn = torch.bmm(q, k.transpose(1,2)) * self.scale  # (B*heads, G, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.bmm(attn, v)  # (B*heads, G, hd)
        ctx = ctx.reshape(B, self.num_heads, self.num_global, self.head_dim).permute(0,2,1,3).reshape(B, self.num_global, self.dim)
        out = self.out_map(ctx)  # B, G, C

        # project global context back into map: simple broadcast addition
        # compute a gating projection from global tokens to per-channel biases
        bias = out.mean(dim=1)[:, :, None, None]  # B, C, 1, 1
        return x + bias  # broadcast as global context


class EfficientAttnBlock(nn.Module):
    """
    Drop-in replacement for AttnBlock with multiple efficient strategies:
      mode = 'window' -> WindowAttention2D(window_size=ws, shift=False/True)
      mode = 'axial'  -> AxialAttention2D
      mode = 'global' -> GlobalPoolAttention (cheap global)
    It preserves residual semantics (returns x + attention(x)).
    """

    def __init__(self, in_channels, mode='window', window_size=8, num_heads=8, shift=False):
        super().__init__()
        self.in_channels = in_channels
        self.mode = mode
        if mode == 'window':
            self.attn = WindowAttention2D(in_channels, num_heads=num_heads, window_size=window_size, shift=shift)
        elif mode == 'axial':
            self.attn = AxialAttention2D(in_channels, num_heads=num_heads)
        elif mode == 'global':
            self.attn = GlobalPoolAttention(in_channels, num_global=8, num_heads=num_heads)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def forward(self, x):
        return self.attn(x)