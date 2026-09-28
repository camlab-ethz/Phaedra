from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Under torchrun, each process gets LOCAL_RANK and must bind to its GPU.
    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is not None:
        try:
            local_rank = int(local_rank_raw)
        except ValueError:
            local_rank = 0

        device_count = torch.cuda.device_count()
        if device_count > 0:
            device_index = local_rank % device_count
            torch.cuda.set_device(device_index)
            return torch.device(f"cuda:{device_index}")

    return torch.device("cuda")


def configure_torch(tf32: bool = True) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)


# ---------------------------------------------------------------------------
# Repository / data / output roots. Every path in the shipped configs is written
# relative to these two environment variables (see env.example.sh).
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_dir(name: str) -> Path:
    v = os.environ.get(name)
    if not v:
        raise EnvironmentError(f"{name} is not set. `source env.example.sh` (after editing it) first.")
    return Path(v).expanduser().resolve()


def data_root() -> Path:
    return _env_dir("PHAEDRA_DATA_ROOT")


def output_root() -> Path:
    return _env_dir("PHAEDRA_OUTPUT_ROOT")


def resolve_repo_path(p: str | os.PathLike) -> Path:
    """Absolute path as-is; relative path -> relative to the repository root."""
    q = Path(str(p)).expanduser()
    return q if q.is_absolute() else (repo_root() / q).resolve()


def portable_path(p: str | os.PathLike) -> str:
    """How a path is recorded in file attributes and metadata: relative to
    $PHAEDRA_OUTPUT_ROOT, $PHAEDRA_DATA_ROOT or the repository when it lies inside one
    of them (e.g. `tokenizers/phaedra_4x4`), else just the file name -- never a
    machine-specific absolute path."""
    q = Path(str(p)).expanduser().resolve()
    roots = [os.environ.get("PHAEDRA_OUTPUT_ROOT"), os.environ.get("PHAEDRA_DATA_ROOT"), str(repo_root())]
    for r in roots:
        if r:
            try:
                return q.relative_to(Path(r).expanduser().resolve()).as_posix()
            except ValueError:
                pass
    return q.name
