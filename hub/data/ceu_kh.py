from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CEUKHDataConfig:
    path: str
    token_type: str
    input_variables: list[str]
    output_variables: list[str]
    split: str
    pair_mode: str
    pair_time_start: int = 0
    pair_time_end: int = 14
    pair_time_step: int = 2
    fixed_input_time: int | None = None
    fixed_output_time: int | None = None
    max_members: int | None = None


def build_time_pairs(
    mode: str,
    time_start: int,
    time_end: int,
    time_step: int,
    fixed_input_time: int | None,
    fixed_output_time: int | None,
    time_len: int,
) -> list[tuple[int, int]]:
    if mode == "all_even_forward":
        if time_step <= 0:
            raise ValueError("pair_time_step must be > 0")
        steps = [t for t in range(time_start, time_end + 1, time_step)]
        pairs: list[tuple[int, int]] = []
        for i, t_in in enumerate(steps[:-1]):
            for t_out in steps[i + 1 :]:
                pairs.append((t_in, t_out))
    elif mode == "fixed":
        if fixed_input_time is None or fixed_output_time is None:
            raise ValueError("fixed pair mode requires fixed_input_time and fixed_output_time")
        pairs = [(int(fixed_input_time), int(fixed_output_time))]
    else:
        raise ValueError(f"Unsupported pair mode: {mode}")

    valid_pairs = []
    for t_in, t_out in pairs:
        if 0 <= t_in < time_len and 0 <= t_out < time_len and t_out > t_in:
            valid_pairs.append((t_in, t_out))
    if not valid_pairs:
        raise ValueError("No valid time pairs were produced")
    return valid_pairs


class CEUKHTokenDataset(Dataset):
    """Shared CEU KelvinHelmholtz dual-token dataset.

    Returns tokens in [V, H, W] format with member/time metadata.
    """

    def __init__(self, cfg: CEUKHDataConfig):
        self.cfg = cfg
        self.token_type = str(cfg.token_type).lower().strip()
        if self.token_type not in {"phaedra", "fsq"}:
            raise ValueError(f"Unsupported token_type: {self.token_type}")
        path = Path(cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.ds = nc.Dataset(str(path), "r")
        self.input_variables = list(cfg.input_variables)
        self.output_variables = list(cfg.output_variables)

        if "time" not in self.ds.dimensions:
            raise ValueError("Token dataset missing time dimension")
        self.time_len = int(self.ds.dimensions["time"].size)

        self.member_values = (
            self.ds.variables["member"][:]
            if "member" in self.ds.variables
            else np.arange(self.ds.dimensions["member"].size)
        )

        split_attr = f"split_{cfg.split}_range"
        if split_attr in self.ds.ncattrs():
            start, end = str(self.ds.getncattr(split_attr)).split(":")
            split_start, split_end = int(start), int(end)
        else:
            split_start, split_end = 0, len(self.member_values)

        self.member_indices = [
            i for i, mid in enumerate(self.member_values) if split_start <= int(mid) < split_end
        ]
        if cfg.max_members is not None:
            self.member_indices = self.member_indices[: int(cfg.max_members)]
        if not self.member_indices:
            raise ValueError(f"No members found for split={cfg.split}")

        if self.token_type == "phaedra":
            self.morph_offset = (
                int(self.ds.getncattr("morphology_offset"))
                if "morphology_offset" in self.ds.ncattrs()
                else 0
            )
        else:
            self.morph_offset = 0

        self._check_required_variables()
        self.time_pairs = build_time_pairs(
            mode=cfg.pair_mode,
            time_start=int(cfg.pair_time_start),
            time_end=int(cfg.pair_time_end),
            time_step=int(cfg.pair_time_step),
            fixed_input_time=cfg.fixed_input_time,
            fixed_output_time=cfg.fixed_output_time,
            time_len=self.time_len,
        )
        self.total_len = len(self.member_indices) * len(self.time_pairs)

    def _check_required_variables(self) -> None:
        needed = set(self.input_variables + self.output_variables)
        missing: list[str] = []
        for var in needed:
            if self.token_type == "phaedra":
                if f"{var}_amp" not in self.ds.variables:
                    missing.append(f"{var}_amp")
                if f"{var}_morph" not in self.ds.variables:
                    missing.append(f"{var}_morph")
            else:
                if f"{var}_tokens" not in self.ds.variables:
                    missing.append(f"{var}_tokens")
        if missing:
            raise ValueError(f"Missing token variables: {missing}")

    def __len__(self) -> int:
        return self.total_len

    def _read_var(self, name: str, member_idx: int, time_idx: int) -> np.ndarray:
        return np.asarray(self.ds.variables[name][member_idx, time_idx], dtype=np.int64)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        member_idx = self.member_indices[idx // len(self.time_pairs)]
        pair_idx = idx % len(self.time_pairs)
        input_time_idx, output_time_idx = self.time_pairs[pair_idx]

        if self.token_type == "phaedra":
            in_morph = []
            in_amp = []
            out_morph = []
            out_amp = []

            for var in self.input_variables:
                in_morph.append(self._read_var(f"{var}_morph", member_idx, input_time_idx))
                in_amp.append(self._read_var(f"{var}_amp", member_idx, input_time_idx))

            for var in self.output_variables:
                out_morph.append(self._read_var(f"{var}_morph", member_idx, output_time_idx))
                out_amp.append(self._read_var(f"{var}_amp", member_idx, output_time_idx))

            in_morph_t = torch.from_numpy(np.stack(in_morph, axis=0)).long()
            out_morph_t = torch.from_numpy(np.stack(out_morph, axis=0)).long()
            if self.morph_offset:
                in_morph_t = in_morph_t - self.morph_offset
                out_morph_t = out_morph_t - self.morph_offset

            return {
                "input_morph": in_morph_t,
                "input_amp": torch.from_numpy(np.stack(in_amp, axis=0)).long(),
                "output_morph": out_morph_t,
                "output_amp": torch.from_numpy(np.stack(out_amp, axis=0)).long(),
                "member_idx": int(self.member_values[member_idx]),
                "input_time_idx": int(input_time_idx),
                "output_time_idx": int(output_time_idx),
                "lead_time_idx": int(output_time_idx - input_time_idx),
                "dataset_name": "ceu_kh",
                "var_names": tuple(self.output_variables),
            }

        in_tokens = []
        out_tokens = []
        for var in self.input_variables:
            in_tokens.append(self._read_var(f"{var}_tokens", member_idx, input_time_idx))
        for var in self.output_variables:
            out_tokens.append(self._read_var(f"{var}_tokens", member_idx, output_time_idx))

        return {
            "input_tokens": torch.from_numpy(np.stack(in_tokens, axis=0)).long(),
            "output_tokens": torch.from_numpy(np.stack(out_tokens, axis=0)).long(),
            "member_idx": int(self.member_values[member_idx]),
            "input_time_idx": int(input_time_idx),
            "output_time_idx": int(output_time_idx),
            "lead_time_idx": int(output_time_idx - input_time_idx),
            "dataset_name": "ceu_kh",
            "var_names": tuple(self.output_variables),
        }

    def close(self) -> None:
        if getattr(self, "ds", None) is not None:
            self.ds.close()
            self.ds = None

    def __del__(self) -> None:
        self.close()
