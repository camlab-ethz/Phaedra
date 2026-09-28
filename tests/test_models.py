"""Every released configuration builds the architecture that was trained: the
parameter counts below are those of the published checkpoints."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("kh", "rc", "rkh")

PARAMS = {  # operator stem -> exact parameter count of the released checkpoints
    "phaedra_38m": 38_059_072,
    "fsq_38m": 37_302_048,
    "vqvae2_38m": 44_300_192,
    "continuous_38m": 38_239_048,
    "fno_38m": 37_833_700,
    "cno_38m": 38_781_664,
    "vit_38m": 37_920_320,
}


def _n(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_operator(model_id: str) -> torch.nn.Module:
    import evaluation.eval_downstream as ed
    reg = OmegaConf.to_container(OmegaConf.load(str(REPO / "evaluation/eval_registry.yaml")), resolve=True)
    entry = reg["models"][model_id]
    fam = entry["family"]
    if fam in ("hub_phaedra", "hub_fsq"):
        from hub.config import load_config
        from hub.models import build_model
        return build_model(load_config(str(REPO / entry["hub_config"])))
    if fam in ("fno", "cno", "vit"):
        m = OmegaConf.to_container(OmegaConf.load(str(REPO / entry["op_config"])), resolve=True)["model"]
        return ed.build_physical_model(fam, m, 4)
    if fam == "continuous":
        from baselines.models_continuous import build_continuous_operator
        return build_continuous_operator(OmegaConf.to_container(OmegaConf.load(str(REPO / entry["train_config"])), resolve=True))
    if fam == "arch_dual_sequential_vq2":
        from baselines.arch.models import BUILDERS
        return BUILDERS["dual_sequential"](ed._arch_config(entry, {}))
    raise AssertionError(fam)


@pytest.mark.parametrize("stem", sorted(PARAMS))
@pytest.mark.parametrize("ds", DATASETS)
def test_operator_param_count(stem, ds):
    assert _n(build_operator(f"{stem}_{ds}")) == PARAMS[stem]


def test_registry_is_consistent():
    reg = OmegaConf.to_container(OmegaConf.load(str(REPO / "evaluation/eval_registry.yaml")), resolve=True)
    assert len(reg["models"]) == 21
    for mid, entry in reg["models"].items():
        cfg = entry.get("hub_config") or entry.get("op_config") or entry.get("train_config")
        assert Path(cfg).stem == mid, (mid, cfg)
        assert Path(entry["checkpoint"]).name == mid
        text = (REPO / cfg).read_text()
        assert f"/operators/{mid}\"" in text, f"{cfg}: output directory is not operators/{mid}"


MAE = ["mae_phaedra_3pde", "mae_fsq_3pde"] + [f"mae_{t}_finetune_{d}" for t in ("phaedra", "fsq") for d in DATASETS]


@pytest.mark.parametrize("mid", MAE)
def test_mae_builds_and_has_test_config(mid):
    from mae.src.build import build_mae
    cfg = OmegaConf.to_container(OmegaConf.load(str(REPO / f"mae/configs/{mid}.yaml")), resolve=True)
    model = build_mae(cfg)
    assert _n(model) > 1_000_000
    test = OmegaConf.load(str(REPO / f"mae/configs/test/{mid}.yaml"))
    assert test.mae_train_config == f"mae/configs/{mid}.yaml"
    assert cfg["validation"]["output_dir"].endswith(f"/mae/{mid}")
