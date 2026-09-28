from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContinuousViTOperatorConfig:
    num_input_vars: int = 4
    num_output_vars: int = 4
    embed_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    patch_size: int = 4
    dropout: float = 0.0
    attn_dropout: float = 0.0
    max_time_index: int = 14
    use_coord_features: bool = True

    @property
    def in_channels(self) -> int:
        return int(self.num_input_vars) + 1 + (2 if bool(self.use_coord_features) else 0)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        hidden_dim = int(float(mlp_ratio) * int(embed_dim))

        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.drop_path1 = nn.Dropout(float(dropout))

        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(embed_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(embed_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path1(h)
        x = x + self.mlp(self.norm2(x))
        return x


class ContinuousViTOperator(nn.Module):
    def __init__(self, cfg: ContinuousViTOperatorConfig):
        super().__init__()
        if int(cfg.depth) <= 0:
            raise ValueError("depth must be > 0")
        if int(cfg.num_input_vars) <= 0 or int(cfg.num_output_vars) <= 0:
            raise ValueError("num_input_vars and num_output_vars must both be > 0")
        if int(cfg.patch_size) <= 0:
            raise ValueError("patch_size must be > 0")
        if int(cfg.embed_dim) % int(cfg.num_heads) != 0:
            raise ValueError(
                f"embed_dim ({cfg.embed_dim}) must be divisible by num_heads ({cfg.num_heads})"
            )
        if int(cfg.embed_dim) % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4 for 2D sin-cos positional encodings")

        self.cfg = cfg
        self.patch_size = int(cfg.patch_size)
        self.max_time_index = max(1, int(cfg.max_time_index))

        self.patch_embed = nn.Conv2d(
            in_channels=int(cfg.in_channels),
            out_channels=int(cfg.embed_dim),
            kernel_size=int(self.patch_size),
            stride=int(self.patch_size),
        )

        self.pos_drop = nn.Dropout(float(cfg.dropout))
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=int(cfg.embed_dim),
                    num_heads=int(cfg.num_heads),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                    attn_dropout=float(cfg.attn_dropout),
                )
                for _ in range(int(cfg.depth))
            ]
        )
        self.norm = nn.LayerNorm(int(cfg.embed_dim))
        self.patch_head = nn.Linear(
            int(cfg.embed_dim),
            int(cfg.num_output_vars) * int(self.patch_size) * int(self.patch_size),
        )

    def _build_features(self, state_norm: torch.Tensor, lead_time_idx: torch.Tensor | int) -> torch.Tensor:
        if state_norm.ndim != 4:
            raise ValueError(f"state_norm must be [B,C,H,W], got {tuple(state_norm.shape)}")

        bsz, channels, height, width = state_norm.shape
        if channels != int(self.cfg.num_input_vars):
            raise ValueError(
                f"Expected {self.cfg.num_input_vars} input variables, got channels={channels}"
            )

        if torch.is_tensor(lead_time_idx):
            lead = lead_time_idx.to(device=state_norm.device, dtype=state_norm.dtype).reshape(-1)
        else:
            lead = torch.tensor([float(lead_time_idx)], device=state_norm.device, dtype=state_norm.dtype)

        if lead.numel() == 1 and bsz > 1:
            lead = lead.expand(bsz)
        if lead.numel() != bsz:
            raise ValueError(
                f"lead_time_idx must contain one value per batch item, got {lead.numel()} for batch={bsz}"
            )

        lead_channel = (lead / float(self.max_time_index)).view(bsz, 1, 1, 1).expand(bsz, 1, height, width)
        features = [state_norm, lead_channel]

        if bool(self.cfg.use_coord_features):
            y = torch.linspace(-1.0, 1.0, height, device=state_norm.device, dtype=state_norm.dtype)
            x = torch.linspace(-1.0, 1.0, width, device=state_norm.device, dtype=state_norm.dtype)
            grid_y = y.view(1, 1, height, 1).expand(bsz, 1, height, width)
            grid_x = x.view(1, 1, 1, width).expand(bsz, 1, height, width)
            features.extend([grid_x, grid_y])

        return torch.cat(features, dim=1)

    @staticmethod
    def _build_2d_sincos_pos_embed(
        grid_h: int,
        grid_w: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        dim_each = int(embed_dim // 4)
        if dim_each <= 0:
            raise ValueError(f"embed_dim must be >= 4, got {embed_dim}")

        yy = torch.arange(int(grid_h), device=device, dtype=torch.float32)
        xx = torch.arange(int(grid_w), device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        grid_x = grid_x.reshape(-1, 1)
        grid_y = grid_y.reshape(-1, 1)

        omega = torch.arange(dim_each, device=device, dtype=torch.float32)
        omega = 1.0 / (10000.0 ** (omega / max(1, dim_each)))

        out_x = grid_x * omega.view(1, -1)
        out_y = grid_y * omega.view(1, -1)

        pos = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1)
        return pos.unsqueeze(0).to(dtype=dtype)

    def forward(self, state_norm: torch.Tensor, lead_time_idx: torch.Tensor | int) -> torch.Tensor:
        x = self._build_features(state_norm, lead_time_idx)
        bsz, _, orig_h, orig_w = x.shape

        pad_h = (self.patch_size - (orig_h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (orig_w % self.patch_size)) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.patch_embed(x)
        grid_h, grid_w = int(x.shape[-2]), int(x.shape[-1])

        x = x.flatten(2).transpose(1, 2)
        pos = self._build_2d_sincos_pos_embed(
            grid_h=grid_h,
            grid_w=grid_w,
            embed_dim=int(self.cfg.embed_dim),
            device=x.device,
            dtype=x.dtype,
        )
        x = self.pos_drop(x + pos)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.patch_head(x)

        x = x.view(
            bsz,
            grid_h,
            grid_w,
            int(self.cfg.num_output_vars),
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(
            bsz,
            int(self.cfg.num_output_vars),
            grid_h * self.patch_size,
            grid_w * self.patch_size,
        )

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :orig_h, :orig_w]
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_parameter_budget(model: nn.Module, target_params: int, tolerance: int) -> None:
    total = count_parameters(model)
    lower = int(target_params) - int(tolerance)
    upper = int(target_params) + int(tolerance)
    if not (lower <= total <= upper):
        raise ValueError(
            f"ViT parameter count {total} is outside required budget [{lower}, {upper}]"
        )
