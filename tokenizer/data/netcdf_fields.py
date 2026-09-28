"""Standalone netCDF field loader for tokenizer training.

Replaces the group-internal `cameleon` data library used by the research code.
It reproduces exactly the sample contract the training systems expect:

    field_variables_in       [C, H, W]  normalized:  (x - mean) / (std + 1e-6)
    field_variables_out      [C, H, W]  raw physical field (same timestep)
    field_variables_in_mean  [C]
    field_variables_in_std   [C]

Data config format (tokenizer/configs/data_*.yaml): a list of datasets, each
one netCDF file + ONE variable (the tokenizers are single-channel), with the
per-variable normalization statistics used everywhere else in the pipeline:

    datasets:
      - name: CEU_2D_KelvinHelmholtzLowRes       # -> <path>/<name>.nc
        path: ${oc.env:PHAEDRA_DATA_ROOT}/fields
        field_variables_in: [rho]
        _normalization_mean: 0.75
        _normalization_std: 0.22776
        max_num_val_samples: 120
        max_num_test_samples: 240

Splits follow the research convention: the LAST `max_num_test_samples`
members are the test set, the `max_num_val_samples` before them the
validation set, everything else training. Every timestep of every member is
one training sample (the tokenizer is a per-snapshot autoencoder).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset


@dataclass
class FieldDatasetSpec:
    name: str
    path: str
    variable: str
    mean: float
    std: float
    max_num_val_samples: int = 120
    max_num_test_samples: int = 240
    num_samples: int | None = None     # cap on members in the split (None = all)


class NetCDFFieldDataset(Dataset):
    """One (member, timestep) snapshot of one variable per item."""

    def __init__(self, spec: FieldDatasetSpec, split: str):
        self.spec = spec
        self.split = split
        self.file = str(Path(spec.path) / f"{spec.name}.nc")
        with nc.Dataset(self.file, "r") as ds:
            var = ds.variables[spec.variable]
            self.n_members = int(ds.dimensions["member"].size)
            self.has_time = "time" in var.dimensions
            self.n_time = int(ds.dimensions["time"].size) if self.has_time else 1
        n_train = self.n_members - spec.max_num_val_samples - spec.max_num_test_samples
        if n_train <= 0:
            raise ValueError(f"{self.file}: {self.n_members} members cannot hold "
                             f"{spec.max_num_val_samples} val + {spec.max_num_test_samples} test")
        starts = {"train": 0, "val": n_train, "test": n_train + spec.max_num_val_samples}
        sizes = {"train": n_train, "val": spec.max_num_val_samples, "test": spec.max_num_test_samples}
        self.member_start = starts[split]
        self.num_members = sizes[split] if spec.num_samples is None else min(sizes[split], int(spec.num_samples))
        self.mean = torch.tensor([float(spec.mean)], dtype=torch.float32)
        self.std = torch.tensor([float(spec.std)], dtype=torch.float32)
        self._handle: nc.Dataset | None = None

    def __len__(self) -> int:
        return self.num_members * self.n_time

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._handle is None:                       # one handle per worker
            self._handle = nc.Dataset(self.file, "r")
        member = self.member_start + idx // self.n_time
        t = idx % self.n_time
        var = self._handle.variables[self.spec.variable]
        raw = np.asarray(var[member, t] if self.has_time else var[member], dtype=np.float32)
        x = torch.from_numpy(raw)[None]               # [1, H, W]
        x_norm = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-6)
        return {
            "field_variables_in": x_norm,
            "field_variables_out": x,
            "field_variables_in_mean": self.mean.clone(),
            "field_variables_in_std": self.std.clone(),
        }


def load_specs(data_config_path: str | Path) -> list[FieldDatasetSpec]:
    raw = OmegaConf.to_container(OmegaConf.load(str(data_config_path)), resolve=True)
    specs = []
    for d in raw["datasets"]:
        variables = list(d.get("field_variables_in") or [])
        if len(variables) != 1:
            raise ValueError("Each dataset entry must list exactly one variable in field_variables_in")
        specs.append(FieldDatasetSpec(
            name=str(d["name"]), path=str(d["path"]), variable=variables[0],
            mean=float(d["_normalization_mean"]), std=float(d["_normalization_std"]),
            max_num_val_samples=int(d.get("max_num_val_samples", 120)),
            max_num_test_samples=int(d.get("max_num_test_samples", 240)),
            num_samples=d.get("num_samples"),
        ))
    return specs


def create_dataloader(data_config_path: str | Path, system_config, max_train_members: int | None = None):
    """Mirror of the research `create_dataloader`: (train_loader, val_loader, [test_loaders])."""
    specs = load_specs(data_config_path)
    train_sets, val_sets, test_sets = [], [], []
    for s in specs:
        s_train = FieldDatasetSpec(**{**s.__dict__, "num_samples": max_train_members})
        train_sets.append(NetCDFFieldDataset(s_train, "train"))
        val_sets.append(NetCDFFieldDataset(s, "val"))
        test_sets.append(NetCDFFieldDataset(s, "test"))
        print(f"[data] {s.name}[{s.variable}] mean={s.mean} std={s.std} "
              f"train={len(train_sets[-1])} val={len(val_sets[-1])} test={len(test_sets[-1])} samples")
    kw = {k: v for k, v in OmegaConf.to_container(system_config.dataloader_kwargs, resolve=True).items() if v is not None}
    kw.pop("shuffle", None)
    train_loader = DataLoader(ConcatDataset(train_sets), shuffle=True, **kw)
    val_loader = DataLoader(ConcatDataset(val_sets), shuffle=True, **kw)
    test_loaders = [DataLoader(t, shuffle=False, **{**kw, "batch_size": 1}) for t in test_sets]
    return train_loader, val_loader, test_loaders
