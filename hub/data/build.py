from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

from .ceu_kh import CEUKHDataConfig, CEUKHTokenDataset
from .collators import build_collator


class EpochShuffleSampler(Sampler[int]):
	"""Deterministic epoch-based sampler that supports mid-epoch resume by batch skipping."""

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
		order = torch.randperm(self.data_len, generator=g).tolist()
		return iter(order)


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
		print(
			f"[data] Capped {label} num_workers from {requested} to {workers} "
			f"(available_cpus={available})"
		)
	return workers


def build_dataloaders(
	cfg: dict,
	*,
	distributed: bool = False,
	rank: int = 0,
	world_size: int = 1,
):
	ds_cfg = cfg["dataset"]
	runtime_seed = int(cfg.get("runtime", {}).get("seed", 42))

	train_ds = CEUKHTokenDataset(
		CEUKHDataConfig(
			path=ds_cfg["path"],
			token_type=str(ds_cfg.get("token_type", "phaedra")),
			input_variables=list(ds_cfg["input_variables"]),
			output_variables=list(ds_cfg["output_variables"]),
			split=str(ds_cfg["split_train"]),
			pair_mode=str(ds_cfg["pair_mode_train"]),
			pair_time_start=int(ds_cfg["time_start"]),
			pair_time_end=int(ds_cfg["time_end"]),
			pair_time_step=int(ds_cfg["time_step"]),
			fixed_input_time=int(ds_cfg["fixed_input_time"]),
			fixed_output_time=int(ds_cfg["fixed_output_time"]),
			max_members=ds_cfg.get("max_train_members"),
		)
	)

	val_ds = CEUKHTokenDataset(
		CEUKHDataConfig(
			path=ds_cfg["path"],
			token_type=str(ds_cfg.get("token_type", "phaedra")),
			input_variables=list(ds_cfg["input_variables"]),
			output_variables=list(ds_cfg["output_variables"]),
			split=str(ds_cfg["split_val"]),
			pair_mode=str(ds_cfg["pair_mode_val"]),
			pair_time_start=int(ds_cfg["time_start"]),
			pair_time_end=int(ds_cfg["time_end"]),
			pair_time_step=int(ds_cfg["time_step"]),
			fixed_input_time=int(ds_cfg["fixed_input_time"]),
			fixed_output_time=int(ds_cfg["fixed_output_time"]),
			max_members=ds_cfg.get("max_val_members"),
		)
	)

	collate_fn = build_collator(cfg["model_type"], ds_cfg, cfg["model"])
	# Drop the ragged final batch only when compiling, so torch.compile sees a
	# single static batch shape and never recompiles at an epoch boundary.
	drop_last_flag = bool(cfg.get("runtime", {}).get("compile", False))
	if distributed and world_size > 1:
		train_sampler: Sampler[int] = DistributedSampler(
			train_ds,
			num_replicas=int(world_size),
			rank=int(rank),
			shuffle=True,
			drop_last=drop_last_flag,
			seed=runtime_seed,
		)
	else:
		train_sampler = EpochShuffleSampler(data_len=len(train_ds), seed=runtime_seed)
	requested_train_workers = int(ds_cfg["num_workers"])
	train_workers = _cap_workers(requested_train_workers, label="train")
	val_requested_workers = max(0, requested_train_workers // 2)
	val_workers = _cap_workers(val_requested_workers, label="val")

	train_loader = DataLoader(
		train_ds,
		batch_size=int(ds_cfg["batch_size_train"]),
		sampler=train_sampler,
		num_workers=train_workers,
		pin_memory=True,
		collate_fn=collate_fn,
		drop_last=drop_last_flag,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=int(ds_cfg["batch_size_val"]),
		shuffle=False,
		num_workers=val_workers,
		pin_memory=True,
		collate_fn=collate_fn,
		drop_last=drop_last_flag,
	)

	return train_loader, val_loader, train_sampler
