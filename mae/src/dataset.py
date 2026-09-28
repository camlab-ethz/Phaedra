from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import List

import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TokenDatasetConfig:
    token_type: str
    path: str | None
    variables: List[str]
    split: str
    paths: List[str] | None = None
    dataset_names: List[str] | None = None


class TokenDataset(Dataset):
    def __init__(self, cfg: TokenDatasetConfig):
        self.cfg = cfg
        self.variables = cfg.variables
        self.token_type = cfg.token_type

        configured_paths = list(cfg.paths) if cfg.paths else ([cfg.path] if cfg.path else [])
        if not configured_paths:
            raise ValueError("TokenDatasetConfig requires either path or paths")

        self.paths = [str(p) for p in configured_paths]
        if cfg.dataset_names is not None:
            if len(cfg.dataset_names) != len(self.paths):
                raise ValueError(
                    f"dataset_names length {len(cfg.dataset_names)} must match paths length {len(self.paths)}"
                )
            self.dataset_names = [str(name) for name in cfg.dataset_names]
        else:
            self.dataset_names = [Path(path).stem for path in self.paths]

        self.datasets = [nc.Dataset(path, "r") for path in self.paths]
        self.dataset_infos: list[dict] = []
        self.sample_offsets: list[int] = []
        self.total_samples = 0

        self.ds = self.datasets[0]
        self.morph_offset = int(self.ds.getncattr("morphology_offset")) if "morphology_offset" in self.ds.ncattrs() else 0

        for dataset_id, ds in enumerate(self.datasets):
            if "time" not in ds.dimensions:
                raise ValueError(f"Token dataset missing time dimension: {self.paths[dataset_id]}")

            time_len = int(ds.dimensions["time"].size)
            member_values = ds.variables["member"][:] if "member" in ds.variables else np.arange(ds.dimensions["member"].size)

            split_attr = f"split_{cfg.split}_range"
            if split_attr in ds.ncattrs():
                start, end = ds.getncattr(split_attr).split(":")
                start = int(start)
                end = int(end)
            else:
                start, end = 0, len(member_values)

            member_indices = [i for i, mid in enumerate(member_values) if start <= mid < end]
            if not member_indices:
                raise ValueError(f"No members found for split {cfg.split} in {self.paths[dataset_id]}")

            self.sample_offsets.append(self.total_samples)
            dataset_samples = len(member_indices) * time_len
            self.total_samples += dataset_samples

            info = {
                "dataset_id": dataset_id,
                "name": self.dataset_names[dataset_id],
                "path": self.paths[dataset_id],
                "ds": ds,
                "time_len": time_len,
                "member_values": member_values,
                "member_indices": member_indices,
                "morph_offset": int(ds.getncattr("morphology_offset")) if "morphology_offset" in ds.ncattrs() else 0,
            }
            self.dataset_infos.append(info)

        self.num_datasets = len(self.dataset_infos)
        self.time_len = int(self.dataset_infos[0]["time_len"])
        self.member_values = self.dataset_infos[0]["member_values"]
        self.member_indices = self.dataset_infos[0]["member_indices"]

    def __len__(self) -> int:
        return self.total_samples

    def get_validation_indices(self, samples_per_dataset: int, time_idx: int) -> list[int]:
        if samples_per_dataset <= 0:
            raise ValueError("samples_per_dataset must be > 0")

        indices: list[int] = []
        for dataset_id, info in enumerate(self.dataset_infos):
            time_len = int(info["time_len"])
            if not (0 <= time_idx < time_len):
                raise ValueError(
                    f"validation.time_index out of range for dataset {info['name']}: "
                    f"{time_idx} not in [0, {time_len - 1}]"
                )
            count = min(samples_per_dataset, len(info["member_indices"]))
            base = self.sample_offsets[dataset_id]
            for pos in range(count):
                indices.append(base + pos * time_len + time_idx)
        return indices

    def _resolve_dataset(self, idx: int) -> tuple[dict, int]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index out of range: {idx}")

        dataset_id = bisect_right(self.sample_offsets, idx) - 1
        local_idx = idx - self.sample_offsets[dataset_id]
        return self.dataset_infos[dataset_id], local_idx

    def __getitem__(self, idx: int):
        info, local_idx = self._resolve_dataset(idx)
        time_len = int(info["time_len"])
        member_indices = info["member_indices"]
        member_values = info["member_values"]
        ds = info["ds"]

        member_idx = member_indices[local_idx // time_len]
        time_idx = local_idx % time_len
        dataset_id = int(info["dataset_id"])

        if self.token_type == "phaedra":
            morph = []
            amp = []
            for var in self.variables:
                morph.append(ds.variables[f"{var}_morph"][member_idx, time_idx])
                amp.append(ds.variables[f"{var}_amp"][member_idx, time_idx])
            morph = torch.from_numpy(np.stack(morph, axis=0)).long()
            morph_offset = int(info["morph_offset"])
            if morph_offset:
                morph = morph - morph_offset
            amp = torch.from_numpy(np.stack(amp, axis=0)).long()
            return {
                "morph": morph,
                "amp": amp,
                "member_idx": int(member_values[member_idx]),
                "time_idx": int(time_idx),
                "dataset_id": dataset_id,
            }

        if self.token_type == "fsq":
            tokens = []
            for var in self.variables:
                tokens.append(ds.variables[f"{var}_tokens"][member_idx, time_idx])
            tokens = torch.from_numpy(np.stack(tokens, axis=0)).long()
            return {
                "tokens": tokens,
                "member_idx": int(member_values[member_idx]),
                "time_idx": int(time_idx),
                "dataset_id": dataset_id,
            }

        raise ValueError(f"Unsupported token_type {self.token_type}")

    def close(self) -> None:
        for info in self.dataset_infos:
            ds = info.get("ds")
            if ds is not None:
                ds.close()
                info["ds"] = None
        self.datasets = []
        self.ds = None
