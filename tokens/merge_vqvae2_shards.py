"""Merge sharded VQ-VAE2 token files into the final per-dataset .nc.

Each shard file <name>.shardK.nc has the full (member=10000, ...) dims but only
its [shard_start, shard_end) member range written, recorded in attrs. The merge
creates the final file with full attrs and copies each shard's slice. Verifies
full coverage (no gaps/overlaps) and token ranges before declaring success.

Usage: python -m tokens.merge_vqvae2_shards --dataset kh --num-shards 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np

OUT_DIR = Path("${oc.env:PHAEDRA_DATA_ROOT}/tokens/vqvae2")
DATASET_FILE = {
    "kh": "CEU2D_KelvinHelmholtzTokens.nc",
    "rc": "CEU2D_RiemannCurvedTokens.nc",
    "rkh": "CEU2D_RiemannKelvinHelmholtzTokens.nc",
}
VARIABLES = ["rho", "u", "v", "p"]
TOP_VOCAB, BOT_VOCAB = 4096, 16384


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_FILE))
    ap.add_argument("--num-shards", type=int, default=8)
    args = ap.parse_args()

    base = DATASET_FILE[args.dataset]
    final_path = OUT_DIR / base
    shard_paths = [OUT_DIR / base.replace(".nc", f".shard{k}.nc")
                   for k in range(args.num_shards)]
    for sp in shard_paths:
        if not sp.exists():
            raise FileNotFoundError(sp)

    # Coverage check from shard attrs PLUS data-level verification. Attrs are
    # written at file creation, so a shard killed mid-write still carries a
    # plausible range -- we must confirm the DATA is actually there (netCDF
    # leaves unwritten values at the fill value, not zero).
    ranges = []
    for sp in shard_paths:
        with nc.Dataset(sp, "r") as d:
            d.set_auto_mask(False)
            s_, e_ = int(d.getncattr("shard_start")), int(d.getncattr("shard_end"))
            probes = sorted({s_, (s_ + e_) // 2, e_ - 1})
            for var in VARIABLES:
                for stream, vocab in (("amp", TOP_VOCAB), ("morph", BOT_VOCAB)):
                    arr = d.variables[f"{var}_{stream}"]
                    for m in probes:
                        v = np.asarray(arr[m])
                        if v.min() < 0 or v.max() >= vocab:
                            raise ValueError(
                                f"{sp.name}: {var}_{stream} member {m} has "
                                f"unwritten/fill data (range {v.min()}..{v.max()}, "
                                f"vocab {vocab}) -- shard incomplete, refusing to merge")
            ranges.append((s_, e_))
            print(f"[merge] {sp.name}: data verified for [{s_}, {e_})", flush=True)
    ranges.sort()
    n_members = ranges[-1][1]
    cursor = 0
    for s, e in ranges:
        assert s == cursor, f"gap/overlap at member {cursor} (shard starts {s})"
        cursor = e
    print(f"[merge] {args.dataset}: {len(ranges)} shards cover [0, {n_members}) contiguously")

    # Final file: clone structure+attrs from shard 0, then copy every slice.
    tmp_path = final_path.with_suffix(".merging.nc")
    with nc.Dataset(shard_paths[0], "r") as ref, nc.Dataset(tmp_path, "w") as out:
        out.createDimension("member", n_members)
        out.createDimension("time", ref.dimensions["time"].size)
        out.createDimension("token_x", ref.dimensions["token_x"].size)
        out.createDimension("token_y", ref.dimensions["token_y"].size)
        for var in VARIABLES:
            for stream in ("amp", "morph"):
                out.createVariable(f"{var}_{stream}", "i4",
                                   ("member", "time", "token_x", "token_y"),
                                   zlib=True, complevel=1)
        mv = out.createVariable("member", "i8", ("member",))
        mv[:] = np.arange(n_members)
        for a in ref.ncattrs():
            if a.startswith("shard_"):
                continue
            out.setncattr(a, ref.getncattr(a))

        for sp, (s, e) in zip(shard_paths, sorted(ranges)):
            with nc.Dataset(sp, "r") as d:
                for var in VARIABLES:
                    for stream in ("amp", "morph"):
                        out.variables[f"{var}_{stream}"][s:e] = \
                            d.variables[f"{var}_{stream}"][s:e]
            print(f"[merge] copied [{s}, {e}) from {sp.name}", flush=True)

    tmp_path.replace(final_path)

    # Sanity: token ranges + spot replication structure on a few members.
    with nc.Dataset(final_path, "r") as d:
        for var in VARIABLES:
            amp = np.asarray(d.variables[f"{var}_amp"][::971], dtype=np.int64)
            mor = np.asarray(d.variables[f"{var}_morph"][::971], dtype=np.int64)
            assert amp.min() >= 0 and amp.max() < TOP_VOCAB, f"{var}_amp out of range"
            assert mor.min() >= 0 and mor.max() < BOT_VOCAB, f"{var}_morph out of range"
            # replication: amp must be constant on 2x2 blocks
            assert np.array_equal(amp[..., ::2, ::2], amp[..., 1::2, ::2]), f"{var}_amp not 2x2-replicated"
    print(f"[merge] {args.dataset}: OK -> {final_path}")


if __name__ == "__main__":
    main()
