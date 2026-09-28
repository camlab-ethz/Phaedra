"""The standalone `phaedra` package and the reproduction code build the same tokenizer:
identical parameter names/shapes, and identical tokens and reconstructions for the same
weights (random weights, CPU). This is what lets `phaedra.load_pretrained()` use the
released checkpoint."""
from __future__ import annotations

import torch


def test_phaedra_package_matches_reproduction_tokenizer(tmp_path):
    from omegaconf import OmegaConf

    import phaedra
    from hub.utils.weights import save_released
    from tokenizer import config_path, system_class
    from tokenizer.pretrained import load_tokenizer

    torch.manual_seed(0)
    task = system_class("Phaedra_AE_FSQ_4x4")(OmegaConf.load(str(config_path("Phaedra_AE_FSQ_4x4"))))
    ref = task.model.state_dict()
    pkg = phaedra.PhaedraModel(OmegaConf.load(phaedra.pretrained.CONFIG_FILE).tokenizer_hyperparameters).state_dict()
    assert {k: tuple(v.shape) for k, v in ref.items()} == {k: tuple(v.shape) for k, v in pkg.items()}

    save_released(ref, tmp_path / "model.safetensors", {"ema": "applied"})
    model = phaedra.load_pretrained(tmp_path)
    tok = load_tokenizer("Phaedra_AE_FSQ_4x4", weights=tmp_path)
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        quant, _, (morph, amp), _ = model.encode(x)
        rec = model.decode(quant)
    amp_ref, morph_ref = tok.encode(x)
    assert torch.equal(morph.reshape(2, 32, 32).long(), morph_ref)
    assert torch.equal(amp.reshape(2, 32, 32).long(), amp_ref)
    assert torch.allclose(rec, tok.decode((amp_ref, morph_ref)), atol=1e-6)
