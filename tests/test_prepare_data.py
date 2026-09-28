"""The Poseidon converter concatenates chunks in NUMERIC order (0,1,2,...,10),
not lexicographically (0,1,10,2,...): this fixes the train/val/test split."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import netCDF4 as nc
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_numeric_chunk_order(tmp_path):
    spec = importlib.util.spec_from_file_location("prep", REPO / "scripts/prepare_poseidon_data.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for k in range(12):                          # data_0 ... data_11, two samples each
        with nc.Dataset(chunks / f"data_{k}.nc", "w") as ds:
            for d, n in (("sample", 2), ("time", 3), ("channel", 5), ("x", 4), ("y", 4)):
                ds.createDimension(d, n)
            v = ds.createVariable("data", "f4", ("sample", "time", "channel", "x", "y"))
            v[:] = np.full((2, 3, 5, 4, 4), k, dtype=np.float32) + np.arange(2)[:, None, None, None, None] * 0.5
    out = tmp_path / "CE.nc"
    prep.convert("CE-KH", chunks, out)
    with nc.Dataset(out) as ds:
        rho = np.asarray(ds.variables["rho"][:, 0, 0, 0])
        assert ds.dimensions["member"].size == 24
    expected = np.array([k + 0.5 * s for k in range(12) for s in range(2)], dtype=np.float32)
    np.testing.assert_array_equal(rho, expected)
