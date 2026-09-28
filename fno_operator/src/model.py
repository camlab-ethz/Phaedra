from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ConditionedFNO2dConfig:
    num_input_vars: int = 4
    num_output_vars: int = 4
    width: int = 96
    depth: int = 8
    modes_x: int = 16
    modes_y: int = 16
    padding: int = 8
    max_time_index: int = 14
    use_coord_features: bool = True

    @property
    def in_channels(self) -> int:
        return int(self.num_input_vars) + 1 + (2 if bool(self.use_coord_features) else 0)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)

        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes_x, self.modes_y)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.modes_x, self.modes_y)
        )

    @staticmethod
    def _complex_multiply(
        x: torch.Tensor,
        weight_real: torch.Tensor,
        weight_imag: torch.Tensor,
    ) -> torch.Tensor:
        xr = x.real.float()
        xi = x.imag.float()
        wr = weight_real.float()
        wi = weight_imag.float()

        out_r = torch.einsum("bixy,ioxy->boxy", xr, wr) - torch.einsum("bixy,ioxy->boxy", xi, wi)
        out_i = torch.einsum("bixy,ioxy->boxy", xr, wi) + torch.einsum("bixy,ioxy->boxy", xi, wr)
        return torch.complex(out_r.float(), out_i.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, size_x, size_y = x.shape
        input_dtype = x.dtype

        x_ft = torch.fft.rfft2(x.float(), norm="ortho")
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            size_x,
            (size_y // 2) + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mx = min(self.modes_x, size_x)
        my = min(self.modes_y, (size_y // 2) + 1)
        out_ft[:, :, :mx, :my] = self._complex_multiply(
            x_ft[:, :, :mx, :my],
            self.weight_real[:, :, :mx, :my],
            self.weight_imag[:, :, :mx, :my],
        )

        out = torch.fft.irfft2(out_ft, s=(size_x, size_y), norm="ortho")
        return out.to(dtype=input_dtype)


class ConditionedFNO2d(nn.Module):
    def __init__(self, cfg: ConditionedFNO2dConfig):
        super().__init__()
        if int(cfg.depth) <= 0:
            raise ValueError("depth must be > 0")
        if int(cfg.num_input_vars) <= 0 or int(cfg.num_output_vars) <= 0:
            raise ValueError("num_input_vars and num_output_vars must both be > 0")

        self.cfg = cfg
        self.padding = int(cfg.padding)
        self.max_time_index = max(1, int(cfg.max_time_index))

        self.lift = nn.Conv2d(int(cfg.in_channels), int(cfg.width), kernel_size=1)
        self.spectral_layers = nn.ModuleList(
            [
                SpectralConv2d(int(cfg.width), int(cfg.width), int(cfg.modes_x), int(cfg.modes_y))
                for _ in range(int(cfg.depth))
            ]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv2d(int(cfg.width), int(cfg.width), kernel_size=1) for _ in range(int(cfg.depth))]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(int(cfg.width), int(cfg.width), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(cfg.width), int(cfg.num_output_vars), kernel_size=1),
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

    def forward(self, state_norm: torch.Tensor, lead_time_idx: torch.Tensor | int) -> torch.Tensor:
        x = self._build_features(state_norm, lead_time_idx)
        x = self.lift(x)

        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))

        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            x = F.gelu(spectral(x) + pointwise(x))

        if self.padding > 0:
            x = x[..., : -self.padding, : -self.padding]

        return self.proj(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def assert_parameter_budget(model: nn.Module, target_params: int, tolerance: int) -> None:
    total = count_parameters(model)
    lower = int(target_params) - int(tolerance)
    upper = int(target_params) + int(tolerance)
    if not (lower <= total <= upper):
        raise ValueError(
            f"FNO parameter count {total} is outside required budget [{lower}, {upper}]"
        )
