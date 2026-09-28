"""CPU-only tests that need no data, no weights and no GPU (run in CI).

Fake token files with the right attributes are written to a temporary
PHAEDRA_DATA_ROOT so that every config can be resolved and every model built.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PROBLEMS = ("KelvinHelmholtz", "RiemannCurved", "RiemannKelvinHelmholtz")


def _fake_tokens(path: Path, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("member", 4)
        ds.createDimension("time", 21)
        ds.createDimension("token_x", 32)
        ds.createDimension("token_y", 32)
        for k, v in attrs.items():
            ds.setncattr(k, v)
        ds.createVariable("member", "i8", ("member",))[:] = np.arange(4)


@pytest.fixture(scope="session", autouse=True)
def roots(tmp_path_factory):
    root = tmp_path_factory.mktemp("phaedra")
    data, out = root / "data", root / "out"
    for p in PROBLEMS:
        _fake_tokens(data / "tokens/phaedra" / f"CEU2D_{p}Tokens.nc",
                     {"amplitude_codebook_size": 1024, "morphology_codebook_size": 8640, "morphology_offset": 1024})
        _fake_tokens(data / "tokens/fsq" / f"CEU2D_{p}Tokens.nc", {"codebook_size": 8640})
        _fake_tokens(data / "tokens/vqvae2" / f"CEU2D_{p}Tokens.nc",
                     {"amplitude_codebook_size": 4096, "morphology_codebook_size": 16384, "morphology_offset": 0})
    out.mkdir()
    os.environ["PHAEDRA_DATA_ROOT"] = str(data)
    os.environ["PHAEDRA_OUTPUT_ROOT"] = str(out)
    os.environ.setdefault("WANDB_MODE", "disabled")
    return {"data": data, "out": out}
