from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ContinuousKHAll2AllConfig:
    token_dataset_path: str
    source_dataset_path: str | None
    split: str
    input_variables: list[str]
    output_variables: list[str]
    pair_mode: str
    pair_time_start: int = 0
    pair_time_end: int = 14
    pair_time_step: int = 2
    fixed_input_time: int = 0
    fixed_output_time: int = 14
    max_members: int | None = None
    train_member_start: int = 0
    train_member_end: int | None = 8000
    val_samples_hint: int | None = None
    test_samples_hint: int | None = None
    normalization: dict[str, dict[str, float]] | None = None


def build_time_pairs(
    mode: str,
    time_start: int,
    time_end: int,
    time_step: int,
    fixed_input_time: int,
    fixed_output_time: int,
    time_len: int,
) -> list[tuple[int, int]]:
    if mode == "all_even_forward":
        if time_step <= 0:
            raise ValueError("pair_time_step must be > 0")
        steps = [t for t in range(int(time_start), int(time_end) + 1, int(time_step))]
        pairs: list[tuple[int, int]] = []
        for i, t_in in enumerate(steps[:-1]):
            for t_out in steps[i + 1 :]:
                pairs.append((int(t_in), int(t_out)))
    elif mode == "fixed":
        pairs = [(int(fixed_input_time), int(fixed_output_time))]
    elif mode == "identity":
        # Steady-state datasets (DAR, POI) have time_len=1 and the input
        # variable differs from the output variable (a -> u, f -> u). One
        # (t, t) pair per available timestep -- normally just (0, 0).
        if time_step <= 0:
            raise ValueError("pair_time_step must be > 0")
        pairs = [(int(t), int(t)) for t in range(int(time_start), int(time_end) + 1, int(time_step))]
    else:
        raise ValueError(f"Unsupported pair mode: {mode}")

    valid_pairs: list[tuple[int, int]] = []
    for t_in, t_out in pairs:
        if not (0 <= int(t_in) < int(time_len) and 0 <= int(t_out) < int(time_len)):
            continue
        # Identity mode allows t_out == t_in; forward modes require t_out > t_in.
        if mode == "identity":
            if int(t_out) == int(t_in):
                valid_pairs.append((int(t_in), int(t_out)))
        else:
            if int(t_out) > int(t_in):
                valid_pairs.append((int(t_in), int(t_out)))

    if not valid_pairs:
        raise ValueError("No valid time pairs were produced")
    return valid_pairs


def _member_values(ds: nc.Dataset) -> np.ndarray:
    if "member" in ds.variables:
        return np.asarray(ds.variables["member"][:], dtype=np.int64)
    if "member" not in ds.dimensions:
        raise ValueError("Continuous dataset is missing 'member' dimension")
    return np.arange(int(ds.dimensions["member"].size), dtype=np.int64)


def _parse_split_range(ds: nc.Dataset, split: str) -> tuple[int, int] | None:
    attr = f"split_{split}_range"
    if attr not in ds.ncattrs():
        return None
    raw = str(ds.getncattr(attr))
    start_s, end_s = raw.split(":")
    return int(start_s), int(end_s)


def _read_member_ids_from_token_split(token_dataset_path: Path, split: str) -> list[int] | None:
    if not token_dataset_path.exists():
        return None

    try:
        with nc.Dataset(str(token_dataset_path), "r") as ds:
            split_range = _parse_split_range(ds, split)
            if split_range is None:
                return None
            start, end = split_range

            values = _member_values(ds)
            ids = [int(mid) for mid in values if int(start) <= int(mid) < int(end)]
            return ids
    except Exception:
        return None


def _fallback_member_slice(
    split: str,
    total_members: int,
    train_member_start: int,
    train_member_end: int | None,
    val_samples_hint: int | None,
    test_samples_hint: int | None,
    source_ds: nc.Dataset,
) -> tuple[int, int]:
    if split == "train":
        start = max(0, int(train_member_start))
        end_default = int(total_members) if train_member_end is None else int(train_member_end)
        end = min(int(total_members), max(start, end_default))
        return start, end

    n_val = int(val_samples_hint) if val_samples_hint is not None else 0
    n_test = int(test_samples_hint) if test_samples_hint is not None else 0

    if n_val <= 0 and "max_num_val_samples" in source_ds.ncattrs():
        n_val = int(source_ds.getncattr("max_num_val_samples"))
    if n_test <= 0 and "max_num_test_samples" in source_ds.ncattrs():
        n_test = int(source_ds.getncattr("max_num_test_samples"))

    train_end = max(0, int(total_members) - int(n_val) - int(n_test))
    val_start = train_end
    val_end = min(int(total_members), val_start + int(n_val))
    test_start = val_end
    test_end = int(total_members)

    if split == "val":
        return val_start, val_end
    if split == "test":
        return test_start, test_end

    raise ValueError(f"Unsupported split: {split}")


def resolve_source_dataset_path(token_dataset_path: Path, source_dataset_path: Path | None) -> Path:
    """Physical-field netCDF for a token file: the configured path, else the
    `source_dataset` attribute of the token file resolved under
    $PHAEDRA_DATA_ROOT/fields."""
    if source_dataset_path is not None:
        resolved = source_dataset_path.expanduser().resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"source_dataset_path does not exist: {resolved}")
    with nc.Dataset(str(token_dataset_path), "r") as ds:
        raw = str(ds.getncattr("source_dataset")) if "source_dataset" in ds.ncattrs() else None
    if not raw:
        raise ValueError(f"{token_dataset_path}: no source_dataset attribute; set dataset.source_dataset_path")
    cands = [Path(raw)]
    data_root = os.environ.get("PHAEDRA_DATA_ROOT")
    if data_root:
        cands.append(Path(data_root) / "fields" / Path(raw).name)
    for c in cands:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"source fields for {token_dataset_path} not found; tried {cands}")


def select_member_indices(
    source_ds: nc.Dataset,
    split: str,
    max_members: int | None,
    token_dataset_path: Path | None,
    train_member_start: int,
    train_member_end: int | None,
    val_samples_hint: int | None,
    test_samples_hint: int | None,
) -> tuple[list[int], list[int]]:
    member_values = _member_values(source_ds)

    # Prefer split ranges from the token dataset to mirror transformer experiments.
    if token_dataset_path is not None:
        token_member_ids = _read_member_ids_from_token_split(token_dataset_path, split)
        if token_member_ids:
            if "member" in source_ds.variables:
                source_lookup = {int(mid): int(i) for i, mid in enumerate(member_values)}
                member_indices = [source_lookup[mid] for mid in token_member_ids if mid in source_lookup]
            else:
                member_indices = [int(mid) for mid in token_member_ids if 0 <= int(mid) < len(member_values)]

            if max_members is not None:
                member_indices = member_indices[: int(max_members)]

            if member_indices:
                member_ids = [int(member_values[i]) for i in member_indices]
                return member_indices, member_ids

    source_split = _parse_split_range(source_ds, split)
    if source_split is not None:
        split_start, split_end = source_split
    else:
        split_start, split_end = _fallback_member_slice(
            split=split,
            total_members=len(member_values),
            train_member_start=train_member_start,
            train_member_end=train_member_end,
            val_samples_hint=val_samples_hint,
            test_samples_hint=test_samples_hint,
            source_ds=source_ds,
        )

    member_indices = [
        i
        for i, member_id in enumerate(member_values)
        if int(split_start) <= int(member_id) < int(split_end)
    ]

    if max_members is not None:
        member_indices = member_indices[: int(max_members)]
    if not member_indices:
        raise ValueError(f"No members selected for split={split}")

    member_ids = [int(member_values[i]) for i in member_indices]
    return member_indices, member_ids


def parse_normalization_stats(
    normalization_cfg: dict[str, dict[str, float]] | None,
    variables: list[str],
) -> dict[str, tuple[float, float]]:
    if normalization_cfg is None:
        raise ValueError("dataset.normalization must be provided")

    stats: dict[str, tuple[float, float]] = {}
    for var_name in variables:
        if var_name not in normalization_cfg:
            raise ValueError(f"Missing normalization stats for variable '{var_name}'")

        entry = normalization_cfg[var_name]
        if not isinstance(entry, dict):
            raise ValueError(f"Normalization entry for '{var_name}' must be a mapping")

        if "mean" not in entry or "std" not in entry:
            raise ValueError(f"Normalization entry for '{var_name}' must define mean and std")

        mean_val = float(entry["mean"])
        std_val = float(entry["std"])
        if std_val == 0.0:
            raise ValueError(f"Normalization std is zero for variable '{var_name}'")

        stats[var_name] = (mean_val, std_val)

    return stats


def read_fields(
    ds: nc.Dataset,
    member_index: int,
    time_idx: int,
    variables: list[str],
) -> np.ndarray:
    # Steady-state datasets store fields as (member, x, y) -- no `time`
    # axis. Time-evolving datasets store (member, time, x, y). Detect from
    # the variable's dim names so the same code path handles both.
    out: list[np.ndarray] = []
    for var_name in variables:
        var = ds.variables[var_name]
        if "time" in var.dimensions:
            arr = np.asarray(var[member_index, time_idx], dtype=np.float32)
        else:
            arr = np.asarray(var[member_index], dtype=np.float32)
        out.append(arr)
    return np.stack(out, axis=0)


def normalize_fields(
    fields: np.ndarray,
    variables: list[str],
    stats: dict[str, tuple[float, float]],
) -> np.ndarray:
    out = np.empty_like(fields, dtype=np.float32)
    for i, var_name in enumerate(variables):
        mean_val, std_val = stats[var_name]
        out[i] = (fields[i] - float(mean_val)) / float(std_val)
    return out


def denormalize_fields(
    fields_norm: np.ndarray,
    variables: list[str],
    stats: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, var_name in enumerate(variables):
        mean_val, std_val = stats[var_name]
        out[var_name] = fields_norm[i] * float(std_val) + float(mean_val)
    return out


class ContinuousKHAll2AllDataset(Dataset):
    def __init__(self, cfg: ContinuousKHAll2AllConfig):
        self.cfg = cfg
        self._source_ds: nc.Dataset | None = None
        self.input_variables = list(cfg.input_variables)
        self.output_variables = list(cfg.output_variables)

        if not self.input_variables or not self.output_variables:
            raise ValueError("input_variables and output_variables must be non-empty")

        self.input_stats = parse_normalization_stats(cfg.normalization, self.input_variables)
        self.output_stats = parse_normalization_stats(cfg.normalization, self.output_variables)

        token_dataset_path = Path(cfg.token_dataset_path).expanduser().resolve()
        source_dataset_path_raw = (
            Path(cfg.source_dataset_path).expanduser().resolve()
            if cfg.source_dataset_path is not None
            else None
        )

        self.token_dataset_path = token_dataset_path
        self.source_dataset_path = resolve_source_dataset_path(
            token_dataset_path=token_dataset_path,
            source_dataset_path=source_dataset_path_raw,
        )

        with nc.Dataset(str(self.source_dataset_path), "r") as source_ds:
            # Steady-state datasets (POI, DAR) have no `time` dim -- treat
            # them as time_len=1 so identity-mode pairs (0, 0) are valid.
            # Per-variable time handling is in `read_fields` below.
            if "time" in source_ds.dimensions:
                self.time_len = int(source_ds.dimensions["time"].size)
            else:
                self.time_len = 1

            missing = [
                name
                for name in set(self.input_variables + self.output_variables)
                if name not in source_ds.variables
            ]
            if missing:
                raise ValueError(f"Missing variables in source dataset: {missing}")

            self.member_indices, self.member_ids = select_member_indices(
                source_ds=source_ds,
                split=str(cfg.split),
                max_members=cfg.max_members,
                token_dataset_path=self.token_dataset_path,
                train_member_start=int(cfg.train_member_start),
                train_member_end=cfg.train_member_end,
                val_samples_hint=cfg.val_samples_hint,
                test_samples_hint=cfg.test_samples_hint,
            )

        self.time_pairs = build_time_pairs(
            mode=str(cfg.pair_mode),
            time_start=int(cfg.pair_time_start),
            time_end=int(cfg.pair_time_end),
            time_step=int(cfg.pair_time_step),
            fixed_input_time=int(cfg.fixed_input_time),
            fixed_output_time=int(cfg.fixed_output_time),
            time_len=self.time_len,
        )

        self.total_len = len(self.member_indices) * len(self.time_pairs)

    def _get_ds(self) -> nc.Dataset:
        if self._source_ds is None:
            self._source_ds = nc.Dataset(str(self.source_dataset_path), "r")
        return self._source_ds

    def close(self) -> None:
        ds = getattr(self, "_source_ds", None)
        if ds is not None:
            ds.close()
            self._source_ds = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.total_len:
            raise IndexError(idx)

        member_pos = idx // len(self.time_pairs)
        pair_pos = idx % len(self.time_pairs)

        member_index = int(self.member_indices[member_pos])
        member_id = int(self.member_ids[member_pos])
        input_time_idx, output_time_idx = self.time_pairs[pair_pos]

        ds = self._get_ds()
        input_raw = read_fields(ds, member_index, int(input_time_idx), self.input_variables)
        target_raw = read_fields(ds, member_index, int(output_time_idx), self.output_variables)

        input_norm = normalize_fields(input_raw, self.input_variables, self.input_stats)
        target_norm = normalize_fields(target_raw, self.output_variables, self.output_stats)

        return {
            "input_fields": torch.from_numpy(input_norm).float(),
            "target_fields": torch.from_numpy(target_norm).float(),
            "lead_time_idx": int(output_time_idx - input_time_idx),
            "input_time_idx": int(input_time_idx),
            "output_time_idx": int(output_time_idx),
            "member_idx": int(member_id),
            "var_names": tuple(self.output_variables),
            "dataset_name": "ceu_kh",
        }
