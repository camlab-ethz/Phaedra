"""Dataset for continuous-latent training: all2all pairs over encoded latents.

Reads the archives written by tokens/encode_continuous_latents.py:
  latents (member, time=8, var=4, ch=8, lh=32, lw=32) float32
  attr time_indices = [0, 2, ..., 14] (original source timesteps)

Pair modes mirror hub/data/ceu_kh.py: `all_even_forward` (28 forward pairs over
the 8 stored steps) and `fixed` (one (t_in, t_out)). Time indices returned in
ORIGINAL units (0..14) so the model's lead-time embedding matches the token
models exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ContinuousLatentDataConfig:
    path: str
    split: str = "train"                  # train | val | test
    pair_mode: str = "all_even_forward"   # all_even_forward | fixed
    fixed_input_time: int = 0
    fixed_output_time: int = 14
    max_members: int | None = None


class ContinuousLatentDataset(Dataset):
    def __init__(self, cfg: ContinuousLatentDataConfig):
        self.cfg = cfg
        self._ds: nc.Dataset | None = None
        path = Path(cfg.path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        self.path = path

        with nc.Dataset(str(path), "r") as ds:
            self.time_indices = [int(t) for t in np.asarray(ds.getncattr("time_indices"))]
            n_members = int(ds.dimensions["member"].size)
            rng = str(ds.getncattr(f"split_{cfg.split}_range")).split(":")
            start, end = int(rng[0]), int(rng[1])
        # member IDs == array indices in the latent archives (written 0..N-1).
        members = [m for m in range(n_members) if start <= m < end]
        if cfg.max_members is not None:
            members = members[: int(cfg.max_members)]
        if not members:
            raise ValueError(f"No members for split={cfg.split} in {path}")
        self.members = members

        t2i = {t: i for i, t in enumerate(self.time_indices)}
        if cfg.pair_mode == "all_even_forward":
            self.pairs = [(a, b) for i, a in enumerate(self.time_indices)
                          for b in self.time_indices[i + 1:]]
        elif cfg.pair_mode == "fixed":
            self.pairs = [(int(cfg.fixed_input_time), int(cfg.fixed_output_time))]
        else:
            raise ValueError(f"pair_mode {cfg.pair_mode!r}")
        self.pair_slots = [(t2i[a], t2i[b]) for a, b in self.pairs]

    def _get(self) -> nc.Dataset:
        if self._ds is None:
            self._ds = nc.Dataset(str(self.path), "r")
        return self._ds

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __len__(self) -> int:
        return len(self.members) * len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        m_pos, p_pos = divmod(idx, len(self.pairs))
        member = self.members[m_pos]
        (t_in, t_out) = self.pairs[p_pos]
        (s_in, s_out) = self.pair_slots[p_pos]
        ds = self._get()
        z_in = np.asarray(ds.variables["latents"][member, s_in], dtype=np.float32)
        z_out = np.asarray(ds.variables["latents"][member, s_out], dtype=np.float32)
        return {
            "input_latents": torch.from_numpy(z_in),     # [V, 8, 32, 32]
            "target_latents": torch.from_numpy(z_out),
            "input_time_idx": int(t_in),
            "output_time_idx": int(t_out),
            "lead_time_idx": int(t_out - t_in),
            "member_idx": int(member),
        }
