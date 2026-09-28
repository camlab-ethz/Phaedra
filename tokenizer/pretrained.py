"""Minimal API for the (pre-)trained tokenizers.

    from tokenizer.pretrained import load_tokenizer
    tok = load_tokenizer("Phaedra_AE_FSQ_4x4")          # released weights from the Hub (or a local path)
    amp, morph = tok.encode(x)                            # x: [B, 1, 128, 128], normalized (see below)
    x_rec = tok.decode((amp, morph))                      # [B, 1, 128, 128], normalized

Inputs/outputs are single-channel fields normalized per variable with the
dataset statistics, x_norm = (x - mean) / std (tables in the model card and in
evaluation/eval_registry.yaml). Token formats:

  Phaedra_AE_FSQ_4x4  (amp [B,32,32] in [0,1024), morph [B,32,32] in [0,8640))
  AE_FSQ              ids [B,32,32] in [0,8640)
  AE_VQVAE2           (top [B,16,16] in [0,4096), bottom [B,32,32] in [0,16384))
  AE_Continuous       latents [B,8,32,32] (float)

The token files on the Hub store Phaedra morphology ids shifted by the
`morphology_offset` attribute (1024) and VQ-VAE-2 top tokens replicated 2x2 onto
the 32x32 grid; use `from_token_file_ids` to undo both.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tokenizer import MODEL_REGISTRY, config_path, system_class


@dataclass
class Tokenizer:
    name: str
    model: torch.nn.Module
    model_type: str
    device: torch.device

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Any:
        x = x.to(self.device, dtype=torch.float32)
        if self.model_type == "continuous":
            return self.model(x, mode="encode")
        _, _, tokens, _ = self.model(x, mode="encode")
        b = x.shape[0]
        if self.model_type == "phaedra":
            morph, amp = tokens
            return _grid(amp, b), _grid(morph, b)
        if self.model_type == "fsq":
            return _grid(tokens, b)
        top, bottom = tokens                                  # vqvae2
        return _grid(top, b), _grid(bottom, b)

    @torch.no_grad()
    def decode(self, tokens: Any) -> torch.Tensor:
        m = self.model
        if self.model_type == "continuous":
            return m(tokens.to(self.device, dtype=torch.float32), mode="decode")
        if self.model_type == "phaedra":
            amp, morph = (t.to(self.device).long() for t in tokens)
            emb = torch.cat([m.quantizer.get_codebook_entry(morph), m.approximate_continuous.get_codebook_entry(amp)], dim=1)
            return m.decode(emb)
        if self.model_type == "fsq":
            return m.decode(m.quantizer.get_codebook_entry(tokens.to(self.device).long()))
        top, bottom = (t.to(self.device).long() for t in tokens)   # vqvae2
        qt = m.quantizer_t.embedding(top.reshape(top.shape[0], -1)).reshape(*top.shape, -1).permute(0, 3, 1, 2)
        qb = m.quantizer_b.embedding(bottom.reshape(bottom.shape[0], -1)).reshape(*bottom.shape, -1).permute(0, 3, 1, 2)
        return m((qt.contiguous(), qb.contiguous()), mode="decode")

    def from_token_file_ids(self, amp_or_top: torch.Tensor, morph_or_bottom: torch.Tensor,
                            morphology_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Undo the storage conventions of the Hub token files (see module docstring)."""
        if self.model_type == "phaedra":
            return amp_or_top.long(), morph_or_bottom.long() - int(morphology_offset)
        if self.model_type == "vqvae2":
            return amp_or_top[..., ::2, ::2].long(), morph_or_bottom.long()
        raise ValueError(f"not applicable to {self.model_type}")


def _grid(idx: torch.Tensor, b: int) -> torch.Tensor:
    idx = idx.reshape(b, -1)
    s = int(round(idx.shape[1] ** 0.5))
    return idx.reshape(b, s, s).long()


def _default_weights(name: str) -> Path | None:
    root = os.environ.get("PHAEDRA_OUTPUT_ROOT")
    if root:
        p = Path(root) / "tokenizers" / MODEL_REGISTRY[name].default_weights
        if p.exists():
            return p
    return None


def _download(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    from hub.utils.hf import MODEL_REPO, REVISION
    sub = MODEL_REGISTRY[name].default_weights
    return Path(hf_hub_download(MODEL_REPO, f"tokenizers/{sub}/model.safetensors", revision=REVISION))


def load_tokenizer(name: str = "Phaedra_AE_FSQ_4x4", weights: str | Path | None = None,
                   device: str | torch.device = "cpu", use_ema: bool = True) -> Tokenizer:
    """Build `name` and load weights from `weights` (file or directory), else
    $PHAEDRA_OUTPUT_ROOT/tokenizers/<id>, else the Hugging Face Hub."""
    from omegaconf import OmegaConf

    from hub.utils.phaedra_decoder import _apply_ema
    from hub.utils.weights import load_into, resolve_weights_file

    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown tokenizer {name}; choose from {sorted(MODEL_REGISTRY)}")
    device = torch.device(device)
    task = system_class(name)(OmegaConf.load(str(config_path(name))))
    src = Path(weights) if weights is not None else (_default_weights(name) or _download(name))
    info = load_into(task.model, src)
    if not info["released"] and use_ema:
        ema = resolve_weights_file(src).parent / "ema.pt"
        if ema.exists():
            _apply_ema(task, ema, torch.device("cpu"))
    model = task.model.to(device).eval()
    return Tokenizer(name=name, model=model, model_type=MODEL_REGISTRY[name].model_type, device=device)
