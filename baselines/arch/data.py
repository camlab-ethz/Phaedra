"""Seq2seq dataloader builder for the benchmark suite.

Reuses `hub.data.ceu_kh.CEUKHTokenDataset` (file-agnostic across CEU2D token
datasets) with `hub.data.collators.Seq2SeqCollator`, which emits the keys
the seq2seq forward expects (`input_amp`, `input_morph`, `input_time_idx`,
`output_time_idx`, etc.) and the matching `output_amp` / `output_morph`
targets.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from hub.data.ceu_kh import CEUKHDataConfig, CEUKHTokenDataset
from hub.data.collators import Seq2SeqCollator


class EpochShuffleSampler(Sampler[int]):
    def __init__(self, data_len: int, seed: int) -> None:
        self.data_len = int(data_len)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.data_len

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.data_len, generator=g).tolist())


def _available_cpu_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, int(os.cpu_count() or 1))


def _cap_workers(requested: int, label: str) -> int:
    requested = max(0, int(requested))
    available = _available_cpu_workers()
    workers = min(requested, available)
    if workers != requested:
        print(f"[data] Capped {label} num_workers {requested} -> {workers} (cpus={available})")
    return workers


def _make_dataset(ds_cfg: dict[str, Any], split: str, pair_mode: str, max_members):
    return CEUKHTokenDataset(
        CEUKHDataConfig(
            path=str(ds_cfg["path"]),
            # hub's CEUKHDataConfig gained a required `token_type` when FSQ
            # support landed; arch-analysis configs predate it.
            token_type=str(ds_cfg.get("token_type", "phaedra")),
            input_variables=list(ds_cfg["input_variables"]),
            output_variables=list(ds_cfg["output_variables"]),
            split=str(split),
            pair_mode=str(pair_mode),
            pair_time_start=int(ds_cfg["time_start"]),
            pair_time_end=int(ds_cfg["time_end"]),
            pair_time_step=int(ds_cfg["time_step"]),
            fixed_input_time=int(ds_cfg["fixed_input_time"]),
            fixed_output_time=int(ds_cfg["fixed_output_time"]),
            max_members=max_members,
        )
    )


def build_seq2seq_dataloaders(
    cfg: dict[str, Any],
    *,
    distributed: bool,
    rank: int,
    world_size: int,
):
    ds_cfg = cfg["dataset"]

    runtime_seed = int(cfg.get("runtime", {}).get("seed", 42))

    train_ds = _make_dataset(ds_cfg, ds_cfg["split_train"], ds_cfg["pair_mode_train"], ds_cfg.get("max_train_members"))
    val_ds = _make_dataset(ds_cfg, ds_cfg["split_val"], ds_cfg["pair_mode_val"], ds_cfg.get("max_val_members"))

    collate = Seq2SeqCollator()

    if distributed and world_size > 1:
        train_sampler: Sampler[int] = DistributedSampler(
            train_ds, num_replicas=int(world_size), rank=int(rank),
            shuffle=True, drop_last=True, seed=runtime_seed,
        )
    else:
        train_sampler = EpochShuffleSampler(data_len=len(train_ds), seed=runtime_seed)

    requested = int(ds_cfg.get("num_workers", 4))
    train_workers = _cap_workers(requested, label="train")
    val_workers = _cap_workers(max(0, requested // 2), label="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(ds_cfg["batch_size_train"]),
        sampler=train_sampler,
        num_workers=train_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=bool(train_workers > 0),
        prefetch_factor=int(ds_cfg.get("prefetch_factor", 4)) if train_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(ds_cfg["batch_size_val"]),
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=bool(val_workers > 0),
        prefetch_factor=int(ds_cfg.get("prefetch_factor", 4)) if val_workers > 0 else None,
    )
    return train_loader, val_loader, train_sampler
