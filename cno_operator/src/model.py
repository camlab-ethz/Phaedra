from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (8, 7, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


def _make_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_norm_groups(int(channels)), int(channels))


@dataclass
class ContinuousCNO2dConfig:
    num_input_vars: int = 4
    num_output_vars: int = 4
    base_width: int = 76
    num_levels: int = 4
    blocks_per_level: int = 2
    bottleneck_blocks: int = 2
    channel_multiplier: int = 2
    dropout: float = 0.0
    max_time_index: int = 14
    use_coord_features: bool = True

    @property
    def in_channels(self) -> int:
        return int(self.num_input_vars) + 1 + (2 if bool(self.use_coord_features) else 0)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1)
        self.norm1 = _make_norm(int(out_channels))
        self.conv2 = nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1)
        self.norm2 = _make_norm(int(out_channels))
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = F.gelu(self.norm1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return F.gelu(x + residual)


class CNOStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int, dropout: float = 0.0):
        super().__init__()
        if int(blocks) <= 0:
            raise ValueError("blocks must be > 0")

        layer_list = [ResidualConvBlock(int(in_channels), int(out_channels), dropout=dropout)]
        for _ in range(int(blocks) - 1):
            layer_list.append(ResidualConvBlock(int(out_channels), int(out_channels), dropout=dropout))
        self.blocks = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ContinuousCNO2d(nn.Module):
    def __init__(self, cfg: ContinuousCNO2dConfig):
        super().__init__()
        if int(cfg.num_levels) < 2:
            raise ValueError("num_levels must be >= 2")
        if int(cfg.blocks_per_level) <= 0:
            raise ValueError("blocks_per_level must be > 0")
        if int(cfg.bottleneck_blocks) <= 0:
            raise ValueError("bottleneck_blocks must be > 0")
        if int(cfg.num_input_vars) <= 0 or int(cfg.num_output_vars) <= 0:
            raise ValueError("num_input_vars and num_output_vars must both be > 0")

        self.cfg = cfg
        self.base_width = int(cfg.base_width)
        self.num_levels = int(cfg.num_levels)
        self.blocks_per_level = int(cfg.blocks_per_level)
        self.bottleneck_blocks = int(cfg.bottleneck_blocks)
        self.channel_multiplier = int(cfg.channel_multiplier)
        self.max_time_index = max(1, int(cfg.max_time_index))

        widths = [int(self.base_width * (self.channel_multiplier**idx)) for idx in range(self.num_levels)]
        self.widths = widths

        self.stem = nn.Conv2d(int(cfg.in_channels), widths[0], kernel_size=1)

        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for level_idx, width in enumerate(widths):
            self.encoder_stages.append(CNOStage(width, width, self.blocks_per_level, dropout=float(cfg.dropout)))
            if level_idx < len(widths) - 1:
                next_width = widths[level_idx + 1]
                self.downsamples.append(
                    nn.Conv2d(width, next_width, kernel_size=3, stride=2, padding=1)
                )

        self.bottleneck = CNOStage(widths[-1], widths[-1], self.bottleneck_blocks, dropout=float(cfg.dropout))

        self.upconvs = nn.ModuleList()
        self.decoder_fuse = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for level_idx in range(len(widths) - 2, -1, -1):
            in_width = widths[level_idx + 1]
            out_width = widths[level_idx]
            self.upconvs.append(nn.ConvTranspose2d(in_width, out_width, kernel_size=2, stride=2))
            self.decoder_fuse.append(nn.Conv2d(out_width * 2, out_width, kernel_size=1))
            self.decoder_stages.append(CNOStage(out_width, out_width, self.blocks_per_level, dropout=float(cfg.dropout)))

        self.head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], kernel_size=1),
            nn.GELU(),
            nn.Conv2d(widths[0], int(cfg.num_output_vars), kernel_size=1),
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
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
        height = int(x.shape[-2])
        width = int(x.shape[-1])
        pad_h = (int(multiple) - (height % int(multiple))) % int(multiple)
        pad_w = (int(multiple) - (width % int(multiple))) % int(multiple)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    @staticmethod
    def _match_spatial(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        target_h = int(reference.shape[-2])
        target_w = int(reference.shape[-1])
        height = int(x.shape[-2])
        width = int(x.shape[-1])
        if height > target_h:
            x = x[:, :, :target_h, :]
        if width > target_w:
            x = x[:, :, :, :target_w]
        if int(x.shape[-2]) < target_h or int(x.shape[-1]) < target_w:
            x = F.pad(x, (0, target_w - int(x.shape[-1]), 0, target_h - int(x.shape[-2])))
        return x

    def forward(self, state_norm: torch.Tensor, lead_time_idx: torch.Tensor | int) -> torch.Tensor:
        x = self._build_features(state_norm, lead_time_idx)
        x, pad = self._pad_to_multiple(x, 2 ** (self.num_levels - 1))
        x = self.stem(x)

        skips: list[torch.Tensor] = []
        for level_idx, encoder_stage in enumerate(self.encoder_stages):
            x = encoder_stage(x)
            if level_idx < len(self.downsamples):
                skips.append(x)
                x = self.downsamples[level_idx](x)

        x = self.bottleneck(x)

        for upconv, fuse, decoder_stage, skip in zip(
            self.upconvs,
            self.decoder_fuse,
            self.decoder_stages,
            reversed(skips),
        ):
            x = upconv(x)
            x = self._match_spatial(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = fuse(x)
            x = decoder_stage(x)

        x = self.head(x)
        pad_h, pad_w = pad
        if pad_h > 0:
            x = x[:, :, : -pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, : -pad_w]
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_parameter_budget(model: nn.Module, target_params: int, tolerance: int) -> None:
    total = count_parameters(model)
    lower = int(target_params) - int(tolerance)
    upper = int(target_params) + int(tolerance)
    if not (lower <= total <= upper):
        raise ValueError(
            f"CNO parameter count {total} is outside required budget [{lower}, {upper}]"
        )
