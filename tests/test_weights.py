"""Weight I/O: released safetensors vs training checkpoints, directory resolution."""
from __future__ import annotations

import torch

from hub.utils.weights import is_released, load_into, load_weights, resolve_weights_file, save_released


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))


def test_safetensors_roundtrip(tmp_path):
    a, b = _net(), torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 2))
    save_released(a.state_dict(), tmp_path / "model.safetensors", {"ema": "applied", "epoch": 3})
    info = load_into(b, tmp_path)                       # directory -> model.safetensors, strict
    assert info["released"] and info["epoch"] == 3 and info["metadata"]["ema"] == "applied"
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])


def test_training_checkpoint_and_priority(tmp_path):
    a = _net()
    torch.save({"model": {f"module.{k}": v for k, v in a.state_dict().items()}, "epoch": 7, "step": 70},
               tmp_path / "checkpoint_last.pt")
    state, info = load_weights(tmp_path)
    assert not info["released"] and info["epoch"] == 7 and set(state) == set(a.state_dict())   # prefix stripped
    save_released(a.state_dict(), tmp_path / "model.safetensors")
    assert resolve_weights_file(tmp_path).name == "model.safetensors" and is_released(tmp_path)
