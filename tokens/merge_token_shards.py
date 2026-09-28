import argparse
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np


def _load_member_ids(ds: nc.Dataset) -> list[int]:
    if "member" in ds.variables:
        return [int(x) for x in ds.variables["member"][:]]
    member_attr = ds.getncattr("member_indices") if "member_indices" in ds.ncattrs() else None
    if member_attr:
        return [int(x) for x in member_attr.split(",") if x]
    raise ValueError("Shard file missing member indices")


def _copy_global_attrs(src: nc.Dataset, dst: nc.Dataset) -> None:
    for k in src.ncattrs():
        if k == "member_indices":
            continue
        dst.setncattr(k, src.getncattr(k))


def _create_dimensions(src: nc.Dataset, dst: nc.Dataset, num_members: int) -> None:
    for dim_name, dim in src.dimensions.items():
        if dim_name == "member":
            dst.createDimension("member", num_members)
        else:
            dst.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)


def _create_variables(src: nc.Dataset, dst: nc.Dataset, token_dtype: str | None) -> None:
    coord_names = {"member", "time", "x", "y", "token_x", "token_y"}
    for var_name, var in src.variables.items():
        dims = var.dimensions
        dtype = var.datatype
        if token_dtype and ("member" in dims) and (var_name not in coord_names) and (var_name not in {"level0_x", "level0_y"}):
            dtype = token_dtype
        dst_var = dst.createVariable(var_name, dtype, dims, zlib=getattr(var, "compression", None) is not None)
        dst_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})


def merge_shards(
    shard_paths: list[Path],
    output_path: Path,
    source_dataset: Path | None,
    token_dtype: str | None,
) -> None:
    shard_paths = sorted(shard_paths)
    if not shard_paths:
        raise ValueError("No shard files provided")

    start_time = time.time()
    print(f"Merging {len(shard_paths)} shard files...")

    # Read member IDs from each shard once so we can validate coverage and duplicates
    shard_member_ids_map: dict[Path, list[int]] = {}
    all_shard_member_ids: list[int] = []
    for shard_path in shard_paths:
        with nc.Dataset(shard_path, "r") as shard_ds:
            ids = _load_member_ids(shard_ds)
        shard_member_ids_map[shard_path] = ids
        all_shard_member_ids.extend(ids)

    if len(all_shard_member_ids) != len(set(all_shard_member_ids)):
        raise ValueError("Duplicate member IDs detected across shards")

    with nc.Dataset(shard_paths[0], "r") as sample_ds:
        coord_cache = {}
        if source_dataset:
            with nc.Dataset(source_dataset, "r") as src_ds:
                num_members = src_ds.dimensions["member"].size
                member_values = src_ds.variables["member"][:] if "member" in src_ds.variables else np.arange(num_members)
                for coord in ("time", "x", "y"):
                    if coord in src_ds.variables:
                        coord_cache[coord] = src_ds.variables[coord][:]
        else:
            # Infer output shape from shard member IDs
            num_members = max(all_shard_member_ids) + 1
            member_values = np.arange(num_members)

        member_values_arr = np.asarray(member_values)
        member_to_output_index = {int(v): idx for idx, v in enumerate(member_values_arr.tolist())}

        # If source_dataset is provided, warn when some members are not present in shards.
        if source_dataset:
            missing_members = set(member_to_output_index.keys()) - set(all_shard_member_ids)
            if missing_members:
                print(
                    f"Warning: {len(missing_members)} members from source dataset are missing in shards; "
                    "their rows will remain unwritten"
                )

        for coord in ("time", "x", "y"):
            if coord not in coord_cache and coord in sample_ds.variables:
                coord_cache[coord] = sample_ds.variables[coord][:]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with nc.Dataset(output_path, "w", format="NETCDF4") as out_ds:
            _create_dimensions(sample_ds, out_ds, num_members)
            _create_variables(sample_ds, out_ds, token_dtype)
            _copy_global_attrs(sample_ds, out_ds)

            # Write coordinate variables
            for coord in ("member", "time", "x", "y"):
                if coord in out_ds.variables:
                    if coord == "member":
                        out_ds.variables[coord][:] = member_values
                    else:
                        target_shape = out_ds.variables[coord].shape
                        coord_vals = coord_cache.get(coord)
                        if coord_vals is not None and coord_vals.shape == target_shape:
                            out_ds.variables[coord][:] = coord_vals
                        else:
                            raise ValueError(
                                f"Coordinate {coord} shape mismatch: target={target_shape}, "
                                f"source={None if coord_vals is None else coord_vals.shape}"
                            )

            # Fill data from shards
            total_members_written = 0
            for shard_idx, shard_path in enumerate(shard_paths, start=1):
                with nc.Dataset(shard_path, "r") as shard_ds:
                    shard_member_ids = shard_member_ids_map[shard_path]
                    for var_name, var in shard_ds.variables.items():
                        if "member" not in var.dimensions:
                            continue
                        member_axis = var.dimensions.index("member")
                        shard_data = var[:]
                        if token_dtype and shard_data.dtype != np.dtype(token_dtype):
                            shard_data = shard_data.astype(token_dtype)
                        for local_idx, member_id in enumerate(shard_member_ids):
                            if member_id not in member_to_output_index:
                                raise ValueError(
                                    f"Shard member ID {member_id} is not present in output member coordinate"
                                )
                            index = [slice(None)] * shard_data.ndim
                            index[member_axis] = local_idx
                            out_index = [slice(None)] * shard_data.ndim
                            out_index[member_axis] = member_to_output_index[member_id]
                            out_ds.variables[var_name][tuple(out_index)] = shard_data[tuple(index)]

                    total_members_written += len(shard_member_ids)
                    elapsed = time.time() - start_time
                    avg_per_member = elapsed / max(total_members_written, 1)
                    remaining = num_members - total_members_written
                    eta = remaining * avg_per_member
                    print(
                        f"[{shard_idx}/{len(shard_paths)}] shard {shard_path.name}: "
                        f"members_written={total_members_written}/{num_members} "
                        f"elapsed={elapsed:.1f}s eta={eta/60:.1f}m"
                    )

    total_time = time.time() - start_time
    print(f"Merge complete in {total_time/60:.1f} minutes -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True, help="Shard .nc files")
    parser.add_argument("--output", required=True, help="Output .nc file")
    parser.add_argument("--source-dataset", default=None, help="Optional source dataset .nc for member coords")
    parser.add_argument("--token-dtype", default=None, choices=["i2", "u2", "i4"], help="Optional token dtype")
    args = parser.parse_args()

    shard_paths = [Path(p) for p in args.shards]
    output_path = Path(args.output)
    source_dataset = Path(args.source_dataset) if args.source_dataset else None

    merge_shards(shard_paths, output_path, source_dataset, args.token_dtype)


if __name__ == "__main__":
    main()

# FSQ
# python -m tokens.merge_token_shards \
#     --shards $PHAEDRA_DATA_ROOT/tokens/fsq/shards/AE_FSQ_tokens_rank*of*.nc \
#     --output $PHAEDRA_DATA_ROOT/tokens/fsq/CEU2D_RiemannKelvinHelmholtzTokens.nc \
#     --source-dataset ${oc.env:PHAEDRA_DATA_ROOT}/fields/CEU_2D_RiemannKelvinHelmholtzLowRes.nc \
#     --token-dtype u2

# Phaedra
# python -m tokens.merge_token_shards \
#     --shards $PHAEDRA_DATA_ROOT/tokens/phaedra/shards/Phaedra_AE_FSQ_4x4_tokens_rank*of*.nc \
#     --output $PHAEDRA_DATA_ROOT/tokens/phaedra/CEU2D_RiemannKelvinHelmholtzTokens.nc \
#     --source-dataset ${oc.env:PHAEDRA_DATA_ROOT}/fields/CEU_2D_RiemannKelvinHelmholtzLowRes.nc \
#     --token-dtype u2