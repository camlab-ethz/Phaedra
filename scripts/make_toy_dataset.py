"""Write small synthetic netCDF files with the CEU_2D_*LowRes schema so the whole
pipeline can be exercised end to end without the real data (scripts/smoke_test.sh).

Schema (identical to the real files): dims member, time=21, x=128, y=128; float32
variables rho, u, v, p (member, time, x, y) plus a `time` coordinate. Fields are
smooth random Fourier modes advected in time -- physically meaningless, but with
the right shapes, dtypes and value ranges for the per-variable statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np

NAMES = ["CEU_2D_KelvinHelmholtzLowRes", "CEU_2D_RiemannCurvedLowRes", "CEU_2D_RiemannKelvinHelmholtzLowRes"]
STATS = {  # (mean, std) per variable, matching the real data statistics in the configs
    "CEU_2D_KelvinHelmholtzLowRes": {"rho": (0.75, 0.22776), "u": (-0.016416, 0.112602), "v": (0.00005, 0.044217), "p": (1.0, 0.0083488)},
    "CEU_2D_RiemannCurvedLowRes": {"rho": (0.548238, 0.322666), "u": (0.00042, 0.275455), "v": (0.0026327, 0.275455), "p": (0.552022, 0.169886)},
    "CEU_2D_RiemannKelvinHelmholtzLowRes": {"rho": (0.54368621, 0.36185354), "u": (-0.00332036, 0.2106914), "v": (0.00215155, 0.215305), "p": (0.5488216, 0.1998227)},
}


def field(rng, n_time, res=128, n_modes=6):
    x = np.linspace(0, 2 * np.pi, res, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    out = np.zeros((n_time, res, res), dtype=np.float32)
    for _ in range(n_modes):
        kx, ky = rng.integers(1, 5, size=2); a = rng.normal(); ph = rng.uniform(0, 2 * np.pi); c = rng.normal(0, 0.3, 2)
        for t in range(n_time):
            out[t] += a * np.sin(kx * (X - c[0] * t * 0.1) + ky * (Y - c[1] * t * 0.1) + ph)
    out /= out.std() + 1e-6
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--members", type=int, default=64)
    ap.add_argument("--time", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for name in NAMES:
        path = out / f"{name}.nc"
        ds = nc.Dataset(path, "w", format="NETCDF4")
        ds.createDimension("member", args.members); ds.createDimension("time", args.time)
        ds.createDimension("x", 128); ds.createDimension("y", 128)
        for d, n in (("member", args.members), ("x", 128), ("y", 128)):
            v = ds.createVariable(d, "i4" if d == "member" else "f4", (d,)); v[:] = np.arange(n)
        tv = ds.createVariable("time", "f4", ("time",)); tv[:] = np.linspace(0.0, 1.0, args.time)
        for var in ("rho", "u", "v", "p"):
            m, s = STATS[name][var]
            v = ds.createVariable(var, "f4", ("member", "time", "x", "y"), zlib=True, complevel=1)
            for mi in range(args.members):
                v[mi] = (m + s * field(rng, args.time)).astype(np.float32)
        ds.close()
        print(f"[toy] wrote {path} ({args.members} members x {args.time} t x 128^2)")


if __name__ == "__main__":
    main()
