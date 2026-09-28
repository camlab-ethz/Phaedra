import argparse
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import sys
import time

import netCDF4 as nc
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from hub.utils.runtime import portable_path
from tokenizer import MODEL_REGISTRY as _TOKENIZER_REGISTRY, config_path as _tok_config, system_class as _tok_system


@dataclass
class ModelSpec:
    task_class: type
    config_path: Path
    model_type: str  # "phaedra" | "fsq"


# Discrete tokenizers that write token grids. (VQ-VAE-2 tokens and continuous
# latents have their own writers: tokens/encode_vqvae2_tokens.py and
# tokens/encode_continuous_latents.py.)
MODEL_REGISTRY = {
    name: ModelSpec(task_class=_tok_system(name), config_path=_tok_config(name),
                    model_type=_TOKENIZER_REGISTRY[name].model_type)
    for name in ("Phaedra_AE_FSQ_4x4", "AE_FSQ")
}


def load_yaml(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if all(k.startswith("model.") for k in keys):
        return {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _load_state_dict_from_path(path: Path) -> dict:
    from hub.utils.weights import load_weights
    return load_weights(path)[0]


def _apply_ema_weights(model: torch.nn.Module, ema_state: dict) -> None:
    shadow = ema_state.get("shadow_params")
    if shadow is None:
        raise ValueError("ema.pt does not contain shadow_params")
    params = list(model.parameters())
    if len(shadow) != len(params):
        raise ValueError("EMA parameter count does not match model parameters")
    for p, s in zip(params, shadow):
        p.data.copy_(s.to(p.device))


def load_model_weights(task, model_path: Path, device: torch.device, use_ema: bool) -> None:
    from hub.utils.weights import is_released
    released = is_released(model_path)        # released safetensors already hold the EMA weights
    state_dict = _load_state_dict_from_path(model_path)
    state_dict = _normalize_state_dict_keys(state_dict)
    missing, unexpected = task.model.load_state_dict(state_dict, strict=released)
    if missing or unexpected:
        print(f"[load] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if use_ema and not released:
        ema_path = model_path / "ema.pt" if model_path.is_dir() else model_path.parent / "ema.pt"
        if not ema_path.exists():
            raise FileNotFoundError(f"ema.pt not found at {ema_path}")
        ema_state = torch.load(ema_path, map_location="cpu")
        _apply_ema_weights(task.model, ema_state)
    task.model.to(device)
    task.model.eval()


def get_split_ranges(num_members: int, max_val: int, max_test: int) -> dict:
    train_end = num_members - max_val - max_test
    return {
        "train": (0, train_end),
        "val": (train_end, train_end + max_val),
        "test": (train_end + max_val, num_members),
    }


def create_output_dataset(
    output_path: Path,
    source_path: Path,
    model_name: str,
    model_path: Path,
    variables: list[str],
    split_ranges: dict,
    model_type: str,
    token_shape: tuple[int, int],
    var_levels: list[tuple[int, int]] | None,
    codebook_meta: dict,
    member_ids: list[int],
    num_timesteps: int,
) -> nc.Dataset:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src = nc.Dataset(source_path, "r")
    out = nc.Dataset(output_path, "w", format="NETCDF4")

    has_time = "time" in src.dimensions

    # Dimensions
    out.createDimension("member", len(member_ids))
    out.createDimension("time", src.dimensions["time"].size if has_time else num_timesteps)
    if "x" in src.dimensions:
        out.createDimension("x", src.dimensions["x"].size)
    if "y" in src.dimensions:
        out.createDimension("y", src.dimensions["y"].size)
    out.createDimension("token_x", token_shape[0])
    out.createDimension("token_y", token_shape[1])

    # Coordinates
    for coord in ("member", "time", "x", "y"):
        if coord in src.variables:
            var = src.variables[coord]
            out_var = out.createVariable(coord, var.datatype, var.dimensions)
            if coord == "member":
                out_var[:] = np.array(member_ids, dtype=var.datatype)
            else:
                out_var[:] = var[:]
            out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})

    if not has_time and "time" not in out.variables:
        out_var = out.createVariable("time", "i4", ("time",))
        out_var[:] = np.arange(num_timesteps, dtype=np.int32)
        out_var.setncattr("note", "synthetic time axis for steady-state dataset")

    out_var = out.createVariable("token_x", "i4", ("token_x",))
    out_var[:] = np.arange(token_shape[0], dtype=np.int32)
    out_var = out.createVariable("token_y", "i4", ("token_y",))
    out_var[:] = np.arange(token_shape[1], dtype=np.int32)

    if model_type == "var" and var_levels:
        for idx, (hx, wx) in enumerate(var_levels):
            out.createDimension(f"level{idx}_x", hx)
            out.createDimension(f"level{idx}_y", wx)
            out_var = out.createVariable(f"level{idx}_x", "i4", (f"level{idx}_x",))
            out_var[:] = np.arange(hx, dtype=np.int32)
            out_var = out.createVariable(f"level{idx}_y", "i4", (f"level{idx}_y",))
            out_var[:] = np.arange(wx, dtype=np.int32)

    # Token variables
    if model_type == "phaedra":
        for var in variables:
            out.createVariable(
                f"{var}_morph",
                "i4",
                ("member", "time", "token_x", "token_y"),
                zlib=True,
            )
            out.createVariable(
                f"{var}_amp",
                "i4",
                ("member", "time", "token_x", "token_y"),
                zlib=True,
            )
    elif model_type == "fsq":
        for var in variables:
            out.createVariable(
                f"{var}_tokens",
                "i4",
                ("member", "time", "token_x", "token_y"),
                zlib=True,
            )
    elif model_type == "var" and var_levels:
        for var in variables:
            for idx, _ in enumerate(var_levels):
                out.createVariable(
                    f"{var}_level{idx}",
                    "i4",
                    ("member", "time", f"level{idx}_x", f"level{idx}_y"),
                    zlib=True,
                )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Global attributes
    out.setncattr("source_dataset", source_path.name)
    out.setncattr("model_name", model_name)
    out.setncattr("model_path", portable_path(model_path))
    out.setncattr("variables", ",".join(variables))
    out.setncattr("split_train_range", f"{split_ranges['train'][0]}:{split_ranges['train'][1]}")
    out.setncattr("split_val_range", f"{split_ranges['val'][0]}:{split_ranges['val'][1]}")
    out.setncattr("split_test_range", f"{split_ranges['test'][0]}:{split_ranges['test'][1]}")
    out.setncattr("member_indices", ",".join(str(mid) for mid in member_ids))
    for k, v in codebook_meta.items():
        out.setncattr(k, v)

    src.close()
    return out


def _reconstruct_from_tokens(task, model_type, tokens, device, amp_codebook_size=None):
    if model_type == "phaedra":
        morph_tokens, amp_tokens = tokens
        morph_tokens = morph_tokens - amp_codebook_size
        morph_tokens = morph_tokens.to(device)
        amp_tokens = amp_tokens.to(device)
        morph_embeddings = task.model.quantizer.get_codebook_entry(morph_tokens)
        amp_embeddings = task.model.approximate_continuous.get_codebook_entry(amp_tokens)
        embeddings = torch.cat([morph_embeddings, amp_embeddings], dim=1)
        recon = task.model.decode(embeddings)
        return recon
    if model_type == "fsq":
        tokens = tokens.to(device)
        embeddings = task.model.quantizer.get_codebook_entry(tokens)
        return task.model.decode(embeddings)
    if model_type == "var":
        raise NotImplementedError("VAR tokenizers are not part of this release")
        for level_tokens in tokens:
            tokens_hier.add_level_batch(level_tokens.to(device))
        max_res = tokens_hier.levels[-1].shape[-1]
        embeddings = task.model.var.reconstruct_from_tokens(tokens_hier, max_res)
        return task.model.decode(embeddings)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _plot_reconstruction_triplets(save_path, inputs, recon_tokens, recon_full, title_prefix):
    rows = len(inputs)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for idx in range(rows):
        row_min = min(inputs[idx].min(), recon_tokens[idx].min(), recon_full[idx].min())
        row_max = max(inputs[idx].max(), recon_tokens[idx].max(), recon_full[idx].max())
        diff = recon_tokens[idx] - recon_full[idx]
        diff_abs_max = np.max(np.abs(diff))

        im0 = axes[idx, 0].imshow(inputs[idx], cmap="turbo", vmin=row_min, vmax=row_max)
        im1 = axes[idx, 1].imshow(recon_tokens[idx], cmap="turbo", vmin=row_min, vmax=row_max)
        im2 = axes[idx, 2].imshow(recon_full[idx], cmap="turbo", vmin=row_min, vmax=row_max)
        im3 = axes[idx, 3].imshow(diff, cmap="bwr", vmin=-diff_abs_max, vmax=diff_abs_max)

        if idx == 0:
            axes[idx, 0].set_title("Input")
            axes[idx, 1].set_title("From Tokens")
            axes[idx, 2].set_title("Full AE")
            axes[idx, 3].set_title(f"Tokens - Full: Max Diff {diff_abs_max:.3f}")

        for ax in axes[idx, :]:
            ax.axis("off")

        fig.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axes[idx, 3], fraction=0.046, pad=0.04)

    fig.suptitle(title_prefix)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True, help="Dataset config yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-members", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-samples", type=int, default=5)
    parser.add_argument("--validate-variable", default="rho")
    parser.add_argument("--validate-time-index", type=int, default=20)
    parser.add_argument("--validate-plot-path", default=None)
    parser.add_argument("--validate-tol", type=float, default=1e-3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--output-name", default=None,
                        help="output file name (default <model-name>_tokens.nc); the pipeline expects "
                             "CEU2D_<Problem>Tokens.nc, e.g. CEU2D_KelvinHelmholtzTokens.nc")
    args = parser.parse_args()

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    dataset_cfg = load_yaml(Path(args.config))["dataset"]
    dataset_name = dataset_cfg["name"]
    dataset_path = Path(dataset_cfg["path"]) / f"{dataset_name}.nc"
    variables = dataset_cfg["variables"]
    num_members = dataset_cfg["num_members"]
    num_timesteps = dataset_cfg["num_timesteps"]
    max_val = dataset_cfg["max_num_val_samples"]
    max_test = dataset_cfg["max_num_test_samples"]
    resolution = tuple(dataset_cfg["resolution"])

    spec = MODEL_REGISTRY[args.model_name]
    model_config = OmegaConf.load(spec.config_path)
    task = spec.task_class(model_config)

    device = torch.device(args.device)
    local_rank_env = os.environ.get("LOCAL_RANK")
    if device.type == "cuda" and local_rank_env is not None:
        torch.cuda.set_device(int(local_rank_env))
        device = torch.device(f"cuda:{int(local_rank_env)}")
    load_model_weights(task, Path(args.model_path), device, args.use_ema)

    split_ranges = get_split_ranges(num_members, max_val, max_test)

    # Determine token shapes and VAR levels with a single forward pass
    sample_member = 0
    sample_var = variables[0]
    with nc.Dataset(dataset_path, "r") as src:
        has_time_dim = "time" in src.variables[sample_var].dimensions
        if has_time_dim:
            sample_data = src.variables[sample_var][sample_member, :, :, :]
        else:
            sample_field = src.variables[sample_var][sample_member, :, :]
            sample_data = np.expand_dims(np.asarray(sample_field), axis=0)
    sample_tensor = torch.from_numpy(sample_data).float().unsqueeze(1)
    mean = dataset_cfg["normalization"][sample_var]["mean"]
    std = dataset_cfg["normalization"][sample_var]["std"]
    sample_tensor = (sample_tensor - mean) / (std + 1e-6)
    sample_tensor = sample_tensor.to(device)

    with torch.inference_mode():
        tokens = task.produce_tokens({"field_variables_in": sample_tensor})

    if spec.model_type == "phaedra":
        morph_tokens, amp_tokens = tokens
        token_shape = (morph_tokens.shape[-2], morph_tokens.shape[-1])
        var_levels = None
        amp_codebook_size = task.model.approximate_continuous.codebook_size
        morph_codebook_size = task.model.quantizer.codebook_size
        codebook_meta = {
            "amplitude_codebook_size": int(amp_codebook_size),
            "morphology_codebook_size": int(morph_codebook_size),
            "morphology_offset": int(amp_codebook_size),
        }
    elif spec.model_type == "fsq":
        token_shape = (tokens.shape[-2], tokens.shape[-1])
        var_levels = None
        codebook_meta = {"codebook_size": int(task.model.quantizer.codebook_size)}
    elif spec.model_type == "var":
        var_levels = [tuple(level.shape[-2:]) for level in tokens.levels]
        token_shape = var_levels[-1]
        codebook_meta = {"codebook_size": int(task.model.var.quantizer.codebook_size)}
    else:
        raise ValueError(f"Unsupported model_type: {spec.model_type}")

    print(f"[token-shape] {token_shape}")

    rank = args.rank
    world_size = args.world_size
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    output_dir = Path(args.output_dir)
    scratch_dir = Path(args.scratch_dir) if args.scratch_dir else None
    if world_size > 1:
        output_name = f"{args.model_name}_tokens_rank{rank}of{world_size}.nc"
    else:
        output_name = args.output_name or f"{args.model_name}_tokens.nc"
    final_output_path = output_dir / output_name
    if scratch_dir:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        output_path = scratch_dir / output_name
    else:
        output_path = final_output_path

    max_member = num_members if args.limit_members is None else min(args.limit_members, num_members)
    shard_members = list(range(rank, max_member, world_size))

    out = create_output_dataset(
        output_path=output_path,
        source_path=dataset_path,
        model_name=args.model_name,
        model_path=Path(args.model_path),
        variables=variables,
        split_ranges=split_ranges,
        model_type=spec.model_type,
        token_shape=token_shape,
        var_levels=var_levels,
        codebook_meta=codebook_meta,
        member_ids=shard_members,
        num_timesteps=num_timesteps,
    )

    start_time = time.time()

    if rank == 0:
        print(f"World Size {world_size}")
        print(f"Shard Members {shard_members}")

    norm_map = dataset_cfg["normalization"]
    with nc.Dataset(dataset_path, "r") as src:
        for local_idx, member_idx in enumerate(shard_members):
            for var in variables:
                var_has_time = "time" in src.variables[var].dimensions
                if var_has_time:
                    # Read the full time axis for a member/variable once to reduce NetCDF call overhead.
                    var_data = src.variables[var][member_idx, :num_timesteps, :, :]
                else:
                    field = src.variables[var][member_idx, :, :]
                    var_data = np.expand_dims(np.asarray(field), axis=0)
                mean = float(norm_map[var]["mean"])
                std = float(norm_map[var]["std"])

                for t_start in range(0, num_timesteps, args.batch_size):
                    t_end = min(t_start + args.batch_size, num_timesteps)
                    batch_np = np.asarray(var_data[t_start:t_end], dtype=np.float32)
                    batch_tensor = torch.from_numpy(batch_np).unsqueeze(1)
                    batch_tensor = (batch_tensor - mean) / (std + 1e-6)
                    batch_tensor = batch_tensor.to(device)

                    with torch.inference_mode():
                        tokens = task.produce_tokens({"field_variables_in": batch_tensor})

                    if spec.model_type == "phaedra":
                        morph_tokens, amp_tokens = tokens
                        morph_tokens = morph_tokens + task.model.approximate_continuous.codebook_size
                        morph_np = morph_tokens.detach().cpu().to(torch.int32).numpy()
                        amp_np = amp_tokens.detach().cpu().to(torch.int32).numpy()
                        out.variables[f"{var}_morph"][local_idx, t_start:t_end, :, :] = morph_np
                        out.variables[f"{var}_amp"][local_idx, t_start:t_end, :, :] = amp_np
                    elif spec.model_type == "fsq":
                        tokens_np = tokens.detach().cpu().to(torch.int32).numpy()
                        out.variables[f"{var}_tokens"][local_idx, t_start:t_end, :, :] = tokens_np
                    elif spec.model_type == "var":
                        level_tokens_np = [level.detach().cpu().to(torch.int32).numpy() for level in tokens.levels]
                        for level_idx, level_np in enumerate(level_tokens_np):
                            out.variables[f"{var}_level{level_idx}"][local_idx, t_start:t_end, :, :] = level_np
                    else:
                        raise ValueError(f"Unsupported model_type: {spec.model_type}")

            if member_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[{args.model_name}] member {member_idx}/{max_member} in {elapsed:.1f}s")

    out.close()

    if scratch_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_path), str(final_output_path))

    if args.validate and rank == 0:
        validate_var = args.validate_variable
        if validate_var not in variables:
            raise ValueError(f"Validation variable {validate_var} not in {variables}")

        plot_path = (
            Path(args.validate_plot_path)
            if args.validate_plot_path
            else output_dir / f"{args.model_name}_{validate_var}_validation.png"
        )

        max_samples = min(args.validate_samples, len(shard_members))
        input_plots = []
        recon_tokens_plots = []
        recon_full_plots = []

        tokens_path = final_output_path if scratch_dir else output_path
        with nc.Dataset(dataset_path, "r") as src, nc.Dataset(tokens_path, "r") as tokens_nc:
            validate_has_time = "time" in src.variables[validate_var].dimensions
            for local_idx, member_idx in enumerate(shard_members[:max_samples]):
                if validate_has_time:
                    data = src.variables[validate_var][member_idx, :, :, :]
                else:
                    field = src.variables[validate_var][member_idx, :, :]
                    data = np.expand_dims(np.asarray(field), axis=0)
                tensor = torch.from_numpy(data).float().unsqueeze(1)
                mean = dataset_cfg["normalization"][validate_var]["mean"]
                std = dataset_cfg["normalization"][validate_var]["std"]
                tensor_norm = (tensor - mean) / (std + 1e-6)
                tensor_norm = tensor_norm.to(device)

                with torch.inference_mode():
                    t_idx = min(args.validate_time_index, tensor_norm.shape[0] - 1)
                    recon_full = task.model(tensor_norm[t_idx:t_idx + 1])[0]

                    if spec.model_type == "phaedra":
                        morph = tokens_nc.variables[f"{validate_var}_morph"][local_idx, t_idx, :, :]
                        amp = tokens_nc.variables[f"{validate_var}_amp"][local_idx, t_idx, :, :]
                        morph_t = torch.from_numpy(morph).to(torch.int64).unsqueeze(0)
                        amp_t = torch.from_numpy(amp).to(torch.int64).unsqueeze(0)
                        recon_tokens = _reconstruct_from_tokens(
                            task,
                            spec.model_type,
                            (morph_t, amp_t),
                            device,
                            amp_codebook_size=task.model.approximate_continuous.codebook_size,
                        )
                    elif spec.model_type == "fsq":
                        tok = tokens_nc.variables[f"{validate_var}_tokens"][local_idx, t_idx, :, :]
                        tok_t = torch.from_numpy(tok).to(torch.int64).unsqueeze(0)
                        recon_tokens = _reconstruct_from_tokens(task, spec.model_type, tok_t, device)
                    else:
                        level_tokens = []
                        for level_idx in range(len(var_levels)):
                            tok = tokens_nc.variables[f"{validate_var}_level{level_idx}"][local_idx, t_idx, :, :]
                            level_tokens.append(torch.from_numpy(tok).to(torch.int64).unsqueeze(0))
                        recon_tokens = _reconstruct_from_tokens(task, spec.model_type, level_tokens, device)

                    diff = (recon_tokens - recon_full).abs().max().item()
                    if diff > args.validate_tol:
                        print(
                            f"Validation failed for member {member_idx}: max abs diff {diff:.6f} > {args.validate_tol}"
                        )
                
                input_plot = data[t_idx, :, :]
                recon_tokens_plot = recon_tokens[0, 0].detach().cpu().numpy() * std + mean
                recon_full_plot = recon_full[0, 0].detach().cpu().numpy() * std + mean
                diff_denorm = float(np.max(np.abs(recon_tokens_plot - recon_full_plot)))
                print(
                    f"Validation stats member {member_idx}: max_abs_norm={diff:.6f} | max_abs_denorm={diff_denorm:.6f}"
                )

                input_plots.append(input_plot)
                recon_tokens_plots.append(recon_tokens_plot)
                recon_full_plots.append(recon_full_plot)

        _plot_reconstruction_triplets(
            plot_path,
            input_plots,
            recon_tokens_plots,
            recon_full_plots,
            f"{args.model_name} | {validate_var} | time={args.validate_time_index}",
        )


if __name__ == "__main__":
    main()
