"""Derive tiny versions of every shipped config (same keys, small models, few
members, one epoch) for the end-to-end smoke test. Writes to
$PHAEDRA_OUTPUT_ROOT/smoke/configs/ and prints the directory.
"""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["PHAEDRA_OUTPUT_ROOT"]) / "smoke" / "configs"
OUT.mkdir(parents=True, exist_ok=True)
DS = ["kh", "rc", "rkh"]
N_VAL, N_TEST = 8, 8


def dump(cfg, name):
    OmegaConf.save(cfg, OUT / name)


# ---- token operators (hub): Phaedra + FSQ
for ds in DS:
    for fam in ("phaedra", "fsq"):
        c = OmegaConf.load(ROOT / "configs/operators" / f"{fam}_38m_{ds}.yaml")
        c.model.update({"embed_dim": 64, "encoder_depth": 2, "num_heads": 4})
        if fam == "phaedra":
            c.model.update({"amp_decoder_depth": 1, "morph_decoder_depth": 1})
        else:
            c.model.update({"decoder_depth": 1})
        c.dataset.update({"max_train_members": 8, "max_val_members": 4, "batch_size_train": 4, "batch_size_val": 2, "num_workers": 2})
        c.training.update({"epochs": 1, "val_every": 20, "checkpoint_every": 1000, "log_every": 5, "warmup_steps": 5})
        c.budget.update({"target_params": 0, "tolerance": 10 ** 9})
        c.validation.update({"max_batches": 1, "max_plots": 1})
        c.testing.update({"max_members": 2, "max_plots": 1, "timing_samples": 2})
        dump(c, f"{fam}_38m_{ds}.yaml")

# ---- physical-space baselines
for ds in DS:
    for fam, model in (("fno", {"width": 16, "depth": 2, "modes_x": 8, "modes_y": 8}),
                       ("cno", {"base_width": 8, "num_levels": 2, "blocks_per_level": 1, "bottleneck_blocks": 1}),
                       ("vit", {"embed_dim": 64, "depth": 2, "num_heads": 4})):
        c = OmegaConf.load(ROOT / "configs/operators" / f"{fam}_38m_{ds}.yaml")
        c.model.update(model)
        c.dataset.update({"max_train_members": 8, "max_val_members": 4, "batch_size_train": 4, "batch_size_val": 2, "num_workers": 2})
        c.training.update({"epochs": 1, "val_every": 20, "checkpoint_every": 1000, "log_every": 5, "warmup_steps": 5})
        c.budget.update({"target_params": 0, "tolerance": 10 ** 9})
        c.validation.update({"max_batches": 1, "max_plots": 1})
        c.testing.update({"max_members": 2, "max_plots": 1, "timing_samples": 2})
        dump(c, f"{fam}_38m_{ds}.yaml")

# ---- continuous-latent transformer
for ds in DS:
    c = OmegaConf.load(ROOT / "configs/operators" / f"continuous_38m_{ds}.yaml")
    c.model.update({"embed_dim": 64, "encoder_depth": 2, "decoder_depth": 1, "num_heads": 4, "grad_checkpointing": False})
    c.dataset.update({"max_train_members": 8, "batch_size_train": 4, "batch_size_val": 2, "num_workers": 2})
    c.training.update({"epochs": 1, "checkpoint_every_epoch": 1, "log_every": 5, "warmup_steps": 5})
    c.budget = {"target_params": None, "tolerance_frac": 1.0}
    c.validation = {"max_batches": 1}
    dump(c, f"continuous_38m_{ds}.yaml")

# ---- VQ-VAE-2 token transformer (dual-decoder trainer)
for ds in DS:
    c = OmegaConf.load(ROOT / "configs/operators" / f"vqvae2_38m_{ds}.yaml")
    c.model.update({"embed_dim": 64, "encoder_depth": 2, "amp_decoder_depth": 1, "morph_decoder_depth": 1, "num_heads": 4})
    c.dataset.update({"max_train_members": 8, "max_val_members": 4, "batch_size_train": 2, "batch_size_val": 2, "num_workers": 2})
    c.training.update({"epochs": 1, "grad_accum_steps": 1, "compile": False, "log_every": 5, "warmup_steps": 5,
                       "checkpoint_every_epoch": 1, "val_every_epoch": 1, "plot_every_epoch": 0})
    c.budget = {"target_params": None, "tolerance_frac": 1.0}
    c.validation.update({"max_batches": 1})
    dump(c, f"vqvae2_38m_{ds}.yaml")

# ---- MAE: 3-PDE pre-training -> RKH fine-tuning (warm start from the run dir) -> test
for mid in ("mae_phaedra_3pde", "mae_phaedra_finetune_rkh"):
    c = OmegaConf.load(ROOT / "mae/configs" / f"{mid}.yaml")
    c.model.update({"embed_dim": 64, "encoder_depth": 2, "decoder_depth": 1, "num_heads": 4})
    c.training.update({"batch_size": 4, "epochs": 1, "log_every": 5, "val_every": 10, "checkpoint_every": 1000,
                       "warmup_steps": 5, "amp_stats_samples": 4})
    c.dataset.update({"copy_to_tmp": False, "num_workers": 2})
    c.validation.update({"samples": 1})
    dump(c, f"{mid}.yaml")
t = OmegaConf.load(ROOT / "mae/configs/test/mae_phaedra_finetune_rkh.yaml")
t.mae_train_config = str(OUT / "mae_phaedra_finetune_rkh.yaml")
t.data_configs = [str(OUT / "tokens_rkh.yaml")]
t.max_test_members = 2
t.max_plots = 1
dump(t, "test_mae_phaedra_finetune_rkh.yaml")

# ---- token-generation configs for the toy files (64 members: 48/8/8)
for ds, src in (("kh", "ceu_kh"), ("rc", "ceu_rc"), ("rkh", "ceu_rkh")):
    c = OmegaConf.load(ROOT / "tokens/configs" / f"{src}.yaml")
    c.dataset.update({"num_members": 64, "max_num_val_samples": N_VAL, "max_num_test_samples": N_TEST})
    dump(c, f"tokens_{ds}.yaml")

# ---- evaluation registry pointing at the smoke configs
reg = OmegaConf.load(ROOT / "evaluation/eval_registry.yaml")
for name, entry in reg.models.items():
    for key in ("hub_config", "op_config", "train_config"):
        if key in entry:
            entry[key] = str(OUT / Path(entry[key]).name)
dump(reg, "eval_registry.yaml")
print(OUT)
