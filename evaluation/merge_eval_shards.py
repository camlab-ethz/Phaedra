"""Merge per-shard evaluator outputs into the canonical CSVs.

`eval_downstream --shard-tag TAG` writes per_sample_relL1.<TAG>.csv (and the
rollout / meta equivalents) so that concurrent eval jobs cannot clobber each
other. This collects them back.

Default is --fresh: the canonical CSV is REBUILT from the shards alone, so a
stale row from an earlier partial run cannot survive into the results table.
Pass --append to merge shards on top of whatever is already there (last writer
wins per model+dataset).

Usage:
  python -m evaluation.merge_eval_shards            # rebuild from shards
  python -m evaluation.merge_eval_shards --append   # keep pre-existing rows
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.eval_downstream import EVAL_ROOT
TAB = EVAL_ROOT / "tables"
CSV_FIELDS = ["model", "dataset", "traj_id", "var", "t", "relL1"]


def _merge_csv(stem: str, fresh: bool) -> None:
    canonical = TAB / f"{stem}.csv"
    shards = sorted(p for p in TAB.glob(f"{stem}.*.csv") if p != canonical)
    if not shards:
        print(f"[merge] {stem}: no shards found")
        return

    # keyed by (model, dataset) so a re-run of one model replaces it wholesale
    groups: dict[tuple, list[dict]] = {}
    if not fresh and canonical.exists():
        with canonical.open() as fh:
            for r in csv.DictReader(fh):
                groups.setdefault((r["model"], r["dataset"]), []).append(r)
    n_shard_rows = 0
    for sp in shards:
        with sp.open() as fh:
            rows = list(csv.DictReader(fh))
        seen = {(r["model"], r["dataset"]) for r in rows}
        for k in seen:                      # shard owns these models outright
            groups[k] = []
        for r in rows:
            groups[(r["model"], r["dataset"])].append(r)
        n_shard_rows += len(rows)
        print(f"[merge] {sp.name}: {len(rows)} rows, {len(seen)} model(s)")

    out = [r for k in sorted(groups) for r in groups[k]]
    canonical.parent.mkdir(parents=True, exist_ok=True)
    tmp = canonical.with_suffix(".merge.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(out)
    tmp.replace(canonical)
    print(f"[merge] -> {canonical.name}: {len(out)} rows from {n_shard_rows} shard rows, "
          f"{len(groups)} model x dataset combos")


def _merge_meta(fresh: bool) -> None:
    canonical = TAB / "per_sample_relL1.meta.json"
    shards = sorted(p for p in TAB.glob("per_sample_relL1.meta.*.json")
                    if p != canonical)
    meta = {} if fresh or not canonical.exists() else json.loads(canonical.read_text())
    for sp in shards:
        meta.update(json.loads(sp.read_text()))
    canonical.write_text(json.dumps(meta, indent=2))
    print(f"[merge] -> {canonical.name}: {len(meta)} entries from {len(shards)} shards")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true",
                    help="keep rows already in the canonical CSV (default: rebuild "
                         "from shards only, so stale rows cannot survive)")
    args = ap.parse_args()
    fresh = not args.append
    print(f"[merge] mode={'append' if args.append else 'fresh rebuild'}")
    _merge_csv("per_sample_relL1", fresh)
    _merge_csv("per_sample_relL1_rollout", fresh)
    _merge_csv("per_sample_relL1_rollout662", fresh)
    _merge_meta(fresh)


if __name__ == "__main__":
    main()
