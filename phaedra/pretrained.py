"""Released Phaedra tokenizer weights for the standalone `phaedra` package.

    from phaedra import load_pretrained
    model = load_pretrained()                    # downloads the NeurIPS 2026 weights from the Hugging Face Hub
    quant, _, (morph, amp), _ = model.encode(x)  # x: [B, 1, 128, 128], normalized per variable
    x_rec = model.decode(quant)

The released weights (EMA) are the tokenizer of the paper, `tokenizers/phaedra_4x4` in
https://huggingface.co/llingsch/phaedra. They load into `PhaedraModel` without any key
changes; `model_config.yaml` in this package holds the matching hyperparameters. Encoding and
decoding are identical to the reproduction code (`tokenizer.pretrained.load_tokenizer`).
"""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

from .phaedra_model import PhaedraModel

MODEL_REPO = os.environ.get("PHAEDRA_HF_MODEL_REPO", "llingsch/phaedra")
REVISION = os.environ.get("PHAEDRA_HF_REVISION", "main")
WEIGHTS_FILE = "tokenizers/phaedra_4x4/model.safetensors"
CONFIG_FILE = Path(__file__).with_name("model_config.yaml")


def load_pretrained(weights: str | os.PathLike | None = None, device: str = "cpu",
                    revision: str | None = None) -> PhaedraModel:
    """Build `PhaedraModel` from `model_config.yaml` and load the released weights.

    Args:
        weights: a local `model.safetensors` (or the folder holding it); default: download
            `tokenizers/phaedra_4x4/model.safetensors` from the Hub (cached by huggingface_hub).
        device: where to put the model (returned in eval mode).
        revision: Hub branch / tag / commit (default: $PHAEDRA_HF_REVISION or main).
    """
    from safetensors.torch import load_file

    if weights is None:
        from huggingface_hub import hf_hub_download
        weights = hf_hub_download(MODEL_REPO, WEIGHTS_FILE, revision=revision or REVISION)
    path = Path(weights)
    if path.is_dir():
        path = path / "model.safetensors"
    model = PhaedraModel(OmegaConf.load(CONFIG_FILE).tokenizer_hyperparameters)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.to(device).eval()
