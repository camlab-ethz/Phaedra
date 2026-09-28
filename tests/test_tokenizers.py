"""All four tokenizers build and round-trip shapes/ranges (random weights, CPU)."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("name", ["Phaedra_AE_FSQ_4x4", "AE_FSQ", "AE_VQVAE2", "AE_Continuous"])
def test_encode_decode_shapes(name, tmp_path):
    from omegaconf import OmegaConf

    from hub.utils.weights import save_released
    from tokenizer import config_path, system_class
    from tokenizer.pretrained import load_tokenizer

    task = system_class(name)(OmegaConf.load(str(config_path(name))))
    save_released(task.model.state_dict(), tmp_path / "model.safetensors", {"ema": "applied"})
    tok = load_tokenizer(name, weights=tmp_path)
    x = torch.randn(1, 1, 128, 128)
    codes = tok.encode(x)
    if name == "Phaedra_AE_FSQ_4x4":
        amp, morph = codes
        assert amp.shape == morph.shape == (1, 32, 32)
        assert 0 <= int(amp.min()) and int(amp.max()) < 1024 and 0 <= int(morph.min()) and int(morph.max()) < 8640
    elif name == "AE_FSQ":
        assert codes.shape == (1, 32, 32) and int(codes.max()) < 8640
    elif name == "AE_VQVAE2":
        assert codes[0].shape == (1, 16, 16) and codes[1].shape == (1, 32, 32)
    else:
        assert codes.shape == (1, 8, 32, 32)
    assert tok.decode(codes).shape == (1, 1, 128, 128)
