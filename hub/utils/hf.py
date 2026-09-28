"""Where the released artifacts live on the Hugging Face Hub.

Override with environment variables (e.g. to use a fork or a pinned revision):
    PHAEDRA_HF_MODEL_REPO   model repository   (tokenizers/, operators/, mae/)
    PHAEDRA_HF_DATA_REPO    dataset repository (tokens/, latents/)
    PHAEDRA_HF_REVISION     branch / tag / commit (default: main)
"""
from __future__ import annotations

import os

MODEL_REPO = os.environ.get("PHAEDRA_HF_MODEL_REPO", "llingsch/phaedra")
DATA_REPO = os.environ.get("PHAEDRA_HF_DATA_REPO", "llingsch/phaedra-tokens")
REVISION = os.environ.get("PHAEDRA_HF_REVISION", "main")
