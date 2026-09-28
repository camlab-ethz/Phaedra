"""Build the field files this project reads from the public Poseidon / PDEgym
datasets (Herde et al., 2024) on the Hugging Face Hub (camlab-ethz/*).

    python scripts/prepare_poseidon_data.py --datasets CE-KH CE-CRP CE-RPUI --download
    python scripts/prepare_poseidon_data.py --datasets CE-KH --chunks-dir /path/to/CE-KH   # already downloaded

writes $PHAEDRA_DATA_ROOT/fields/<name>.nc, e.g. CEU_2D_KelvinHelmholtzLowRes.nc.

| public dataset | written as                             | used for                         |
|----------------|----------------------------------------|----------------------------------|
| CE-KH          | CEU_2D_KelvinHelmholtzLowRes.nc        | operators / MAE ("KH"), tokenizer |
| CE-CRP         | CEU_2D_RiemannCurvedLowRes.nc          | operators / MAE ("RC"), tokenizer |
| CE-RPUI        | CEU_2D_RiemannKelvinHelmholtzLowRes.nc | operators / MAE ("RKH")           |
| CE-RP          | CEU_2D_RiemannLowRes.nc                | tokenizer pre-training            |
| CE-Gauss       | CEU_2D_GaussLowRes.nc                  | tokenizer pre-training            |
| NS-Gauss       | IEU_2D_Gauss.nc                        | tokenizer pre-training            |
| NS-Sines       | IEU_2D_Sin.nc                          | tokenizer pre-training            |

IMPORTANT -- trajectory order. The public datasets ship as chunks data_0.nc ...
data_13.nc. They must be concatenated in NUMERIC chunk order (0, 1, 2, ..., 13);
this is how the files used in the paper were assembled, and it fixes the
train/val/test split (the last 240 trajectories are the test set). Poseidon's own
`assemble_data.py` sorts file names lexicographically (0, 1, 10, 11, 12, 13, 2, ...),
which yields a different trajectory order -- do not use it for this project.

Every converted file is checked against sha256 hashes of selected trajectories of
the files used in the paper (--no-verify to skip).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

# public id -> (output name, variable names in channel order, member dtype, x/y grid, chunked output)
DATASETS = {
    "CE-KH": ("CEU_2D_KelvinHelmholtzLowRes", ["rho", "u", "v", "p", "E"], "i4", "endpoint", False),
    "CE-CRP": ("CEU_2D_RiemannCurvedLowRes", ["rho", "u", "v", "p", "E"], "i4", "endpoint", False),
    "CE-RPUI": ("CEU_2D_RiemannKelvinHelmholtzLowRes", ["rho", "u", "v", "p", "E"], "i4", "endpoint", False),
    "CE-RP": ("CEU_2D_RiemannLowRes", ["rho", "u", "v", "p", "E"], "i4", "endpoint", False),
    "CE-Gauss": ("CEU_2D_GaussLowRes", ["rho", "u", "v", "p", "E"], "i4", "endpoint", False),
    "NS-Gauss": ("IEU_2D_Gauss", ["u", "v", "rho"], "i8", "periodic", True),     # rho = passive tracer
    "NS-Sines": ("IEU_2D_Sin", ["u", "v", "rho"], "i8", "periodic", True),
}

# sha256 of the raw float32 bytes of <variable>[member] (all 21 timesteps) in the
# files used for the paper -- filled from the release audit (evaluation/data_hashes.json).
REFERENCE_HASHES_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "data_hashes.json"


def chunk_files(folder: Path) -> list[Path]:
    """data_<k>.nc / velocity_<k>.nc sorted by the NUMERIC chunk index."""
    files = [p for p in folder.glob("*.nc") if re.fullmatch(r"[A-Za-z]+_\d+\.nc", p.name)]
    if not files:
        raise FileNotFoundError(f"no chunk files (e.g. data_0.nc) in {folder}")
    return sorted(files, key=lambda p: int(re.search(r"_(\d+)\.nc$", p.name).group(1)))


def download(dataset: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download
    print(f"[prepare] downloading camlab-ethz/{dataset} -> {dest} (~70-85 GB)", flush=True)
    snapshot_download(repo_id=f"camlab-ethz/{dataset}", repo_type="dataset", local_dir=str(dest),
                      allow_patterns=["*.nc"])
    return dest


def convert(dataset: str, chunks_dir: Path, out_path: Path, block: int = 64) -> None:
    name, variables, member_dtype, grid, chunked = DATASETS[dataset]
    files = chunk_files(chunks_dir)
    sizes = []
    for f in files:
        with nc.Dataset(f, "r") as ds:
            sizes.append(ds.dimensions["sample"].size)
            n_time = ds.dimensions["time"].size
            nx, ny = ds.dimensions["x"].size, ds.dimensions["y"].size
    n = int(sum(sizes))
    print(f"[prepare] {dataset}: {len(files)} chunks (numeric order), {n} trajectories -> {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".nc.tmp")
    out = nc.Dataset(tmp, "w", format="NETCDF4")
    out.createDimension("member", n)
    out.createDimension("time", n_time)
    out.createDimension("x", nx)
    out.createDimension("y", ny)
    out.createVariable("member", member_dtype, ("member",))[:] = np.arange(n)
    if grid == "endpoint":            # CE files: x = linspace(0, 1, 128) incl. both ends
        xs, ys = np.linspace(0, 1, nx, dtype=np.float32), np.linspace(0, 1, ny, dtype=np.float32)
    else:                             # NS files: periodic grid x = i / 128
        xs, ys = (np.arange(nx) / nx).astype(np.float32), (np.arange(ny) / ny).astype(np.float32)
    out.createVariable("x", "f4", ("x",))[:] = xs
    out.createVariable("y", "f4", ("y",))[:] = ys
    out.createVariable("time", "f4", ("time",))[:] = np.linspace(0, 1, n_time, dtype=np.float32)
    kw = {"chunksizes": (1, 1, nx, ny)} if chunked else {}
    outv = {v: out.createVariable(v, "f4", ("member", "time", "x", "y"), **kw) for v in variables}
    lo = 0
    for f, sz in zip(files, sizes):
        with nc.Dataset(f, "r") as ds:
            ds.set_auto_mask(False)
            src = ds.variables[f.name.split("_")[0]]          # (sample, time, channel, x, y)
            for b in range(0, sz, block):
                e = min(sz, b + block)
                arr = np.asarray(src[b:e], dtype=np.float32)
                for c, v in enumerate(variables):
                    outv[v][lo + b:lo + e] = arr[:, :, c]
        lo += sz
        print(f"[prepare]   {f.name}: {sz} trajectories -> members [{lo - sz}, {lo})", flush=True)
    out.close()
    os.replace(tmp, out_path)


def verify(dataset: str, out_path: Path) -> bool:
    import json
    name = DATASETS[dataset][0]
    if not REFERENCE_HASHES_FILE.exists():
        print("[prepare] no reference hashes shipped; skipping verification", flush=True)
        return True
    ref = json.loads(REFERENCE_HASHES_FILE.read_text()).get(name)
    if ref is None:
        print(f"[prepare] no reference hashes for {name}; skipping verification", flush=True)
        return True
    ok = True
    with nc.Dataset(out_path, "r") as ds:
        ds.set_auto_mask(False)
        if ds.dimensions["member"].size != ref["n_members"]:
            print(f"[prepare] FAIL {name}: {ds.dimensions['member'].size} members, expected {ref['n_members']}")
            return False
        for m, h in ref["members"].items():
            got = hashlib.sha256(np.ascontiguousarray(ds.variables[ref["variable"]][int(m)], dtype=np.float32).tobytes()).hexdigest()
            if got != h:
                ok = False
                print(f"[prepare] FAIL {name}: member {m} differs from the paper's file", flush=True)
    print(f"[prepare] {name}: {'matches' if ok else 'DOES NOT MATCH'} the files used in the paper "
          f"({len(ref['members'])} reference trajectories)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["CE-KH", "CE-CRP", "CE-RPUI"], choices=sorted(DATASETS))
    ap.add_argument("--chunks-dir", default=None,
                    help="folder with the downloaded chunks (one dataset), or a parent folder containing "
                         "one sub-folder per dataset id (default: $PHAEDRA_DATA_ROOT/poseidon/<id>)")
    ap.add_argument("--download", action="store_true", help="download missing datasets from the Hugging Face Hub")
    ap.add_argument("--out-dir", default=None, help="default: $PHAEDRA_DATA_ROOT/fields")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    data_root = Path(os.environ.get("PHAEDRA_DATA_ROOT", "."))
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "fields"
    failed = []
    for d in args.datasets:
        if args.chunks_dir and len(args.datasets) == 1 and any(Path(args.chunks_dir).glob("*_0.nc")):
            cdir = Path(args.chunks_dir)
        else:
            cdir = (Path(args.chunks_dir) if args.chunks_dir else data_root / "poseidon") / d
        if args.download and not any(cdir.glob("*_0.nc")):
            download(d, cdir)
        out_path = out_dir / f"{DATASETS[d][0]}.nc"
        convert(d, cdir, out_path)
        if not args.no_verify and not verify(d, out_path):
            failed.append(d)
    if failed:
        sys.exit(f"[prepare] verification failed for {failed}")


if __name__ == "__main__":
    main()
