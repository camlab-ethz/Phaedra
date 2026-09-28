"""Train a tokenizer (Phaedra 4x4, or the FSQ / VQ-VAE-2 / continuous-AE baselines).

    accelerate launch --num_processes 4 -m tokenizer.train --model Phaedra_AE_FSQ_4x4
    python -m tokenizer.train --model AE_FSQ --data tokenizer/configs/data_pretraining.yaml

The model config (hyper-parameters, optimizer, output directories) comes from
tokenizer/configs/model_*.yaml (override with --config); the data config lists
the netCDF files + normalization statistics (see tokenizer/data/netcdf_fields.py).
Checkpoints are written with `accelerate.save_state` to
`<checkpoint_dir>/<experiment_name>_<step>/` (pytorch_model.bin + ema.pt), which is
exactly the directory layout every downstream script loads (`--model-path`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from tokenizer import MODEL_REGISTRY, config_path, system_class
from tokenizer.data.netcdf_fields import create_dataloader
from tokenizer.train_loop import fit

DEFAULT_DATA = Path(__file__).resolve().parent / "configs" / "data_pretraining.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--config", default=None, help="model yaml (default: packaged config for --model)")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="data yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--max-train-members", type=int, default=None, help="cap training members per dataset (smoke tests)")
    ap.add_argument("--experiment-name", default=None)
    ap.add_argument("--export-dir", default=None,
                    help="copy the FINAL checkpoint (pytorch_model.bin + ema.pt) here; default "
                         "$PHAEDRA_OUTPUT_ROOT/tokenizers/<model> = where every downstream config looks. "
                         "Pass 'none' to skip.")
    ap.add_argument("--output-root", default=None,
                    help="override logging/checkpoint/test dirs to <output-root>/{logs,checkpoints,test}")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config or str(config_path(args.model)))
    hp = cfg.training_hyperparameters
    if args.epochs is not None:
        hp.epochs = int(args.epochs)
    if args.max_steps is not None:
        hp.max_steps = int(args.max_steps)
    if args.experiment_name:
        hp.experiment_name = args.experiment_name
    if args.output_root:
        root = Path(args.output_root)
        hp.logging_dir, hp.checkpoint_dir, hp.test_dir = str(root / "logs"), str(root / "checkpoints"), str(root / "test")

    task = system_class(args.model)(cfg)
    train_loader, val_loader, _ = create_dataloader(args.data, cfg.system_config, max_train_members=args.max_train_members)
    print(f"[train] model={args.model} params={sum(p.numel() for p in task.parameters()):,} "
          f"train_batches/epoch={len(train_loader)}")
    fit(task, cfg, train_loader=train_loader, val_loader=val_loader)
    _export_final(cfg, args)


def _export_final(cfg, args) -> None:
    """Copy the last saved checkpoint dir to the canonical weights location."""
    import os
    import re
    import shutil
    if args.export_dir == "none":
        return
    hp = cfg.training_hyperparameters
    ckpt_dir = Path(str(hp.checkpoint_dir))
    cands = [d for d in ckpt_dir.glob(f"{hp.experiment_name}_*") if d.is_dir() and re.fullmatch(r".*_\d+", d.name)]
    if not cands:
        print(f"[export] no checkpoint dirs under {ckpt_dir}; nothing exported")
        return
    last = max(cands, key=lambda d: int(d.name.rsplit("_", 1)[1]))
    dest = Path(args.export_dir) if args.export_dir else \
        Path(os.environ["PHAEDRA_OUTPUT_ROOT"]) / "tokenizers" / MODEL_REGISTRY[args.model].default_weights
    if os.environ.get("RANK", "0") not in ("0", ""):
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("pytorch_model.bin", "ema.pt"):
        src = last / name
        if src.exists():
            shutil.copy2(src, dest / name)
    (dest / "SOURCE.txt").write_text(f"{last}\n")
    print(f"[export] {last.name} -> {dest}")


if __name__ == "__main__":
    main()
