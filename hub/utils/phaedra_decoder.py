from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


@dataclass
class PhaedraDecoderHandle:
    task: Any
    device: torch.device


@dataclass
class TokenDecoderHandle:
    task: Any
    device: torch.device
    model_type: str


def _normalize_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    if keys and all(k.startswith("model.") for k in keys):
        return {k.replace("model.", "", 1): v for k, v in state.items()}
    return state


def _load_state_dict(path: Path) -> dict[str, Any]:
    """Weights from released `model.safetensors`, `pytorch_model.bin`, or a checkpoint file/dir."""
    from hub.utils.weights import load_weights
    return load_weights(path)[0]


def _apply_ema(task: Any, path: Path, device: torch.device) -> None:
    ema_state = torch.load(path, map_location="cpu")
    shadow = ema_state.get("shadow_params")
    if shadow is None:
        raise ValueError(f"EMA file missing shadow_params: {path}")

    params = list(task.model.parameters())
    if len(params) != len(shadow):
        raise ValueError("EMA shadow length does not match model parameter length")

    for p, s in zip(params, shadow):
        p.data.copy_(s.to(device))


def _resolve_model_config_path(model_name: str, raw_cfg: dict[str, Any]) -> Path:
    override = raw_cfg.get("model_config")
    if override:
        return Path(str(override)).expanduser().resolve()
    from tokenizer import config_path
    return config_path(model_name)


def load_token_decoder(raw_cfg: dict[str, Any], device: torch.device) -> TokenDecoderHandle | None:
    """Load a trained tokenizer for decoding.

    raw_cfg keys: enabled, model_name (Phaedra_AE_FSQ_4x4 | AE_FSQ | AE_VQVAE2 |
    AE_Continuous), model_path (checkpoint dir with pytorch_model.bin [+ ema.pt],
    or a file), use_ema, optional model_config (yaml override). The legacy
    `phaedra_root` key is ignored: the tokenizer code is vendored in `tokenizer/`.
    """
    if not raw_cfg.get("enabled", False):
        return None
    from tokenizer import MODEL_REGISTRY, system_class

    model_name = str(raw_cfg.get("model_name", "Phaedra_AE_FSQ_4x4"))
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported decoder model_name={model_name}; known: {sorted(MODEL_REGISTRY)}")
    task_class = system_class(model_name)
    model_type = MODEL_REGISTRY[model_name].model_type

    model_cfg_path = _resolve_model_config_path(model_name, raw_cfg)
    model_path = Path(raw_cfg["model_path"]).expanduser().resolve()
    ema_override = None
    if model_path.is_file() and model_path.name == "ema.pt":
        ema_override = model_path
        model_path = model_path.parent

    from hub.utils.weights import is_released
    released = is_released(model_path)        # safetensors: EMA already baked in
    task = task_class(OmegaConf.load(model_cfg_path))
    state_dict = _normalize_state_dict(_load_state_dict(model_path))
    task.model.load_state_dict(state_dict, strict=released)

    if bool(raw_cfg.get("use_ema", False)) and not released:
        ema_path = ema_override or (model_path / "ema.pt" if model_path.is_dir() else model_path.parent / "ema.pt")
        if not ema_path.exists():
            raise FileNotFoundError(f"EMA file not found: {ema_path}")
        _apply_ema(task, ema_path, device)

    task.model.to(device)
    task.model.eval()

    return TokenDecoderHandle(task=task, device=device, model_type=model_type)


def load_phaedra_decoder(raw_cfg: dict[str, Any], device: torch.device) -> PhaedraDecoderHandle | None:
    handle = load_token_decoder(raw_cfg, device=device)
    if handle is None:
        return None
    if handle.model_type != "phaedra":
        raise ValueError("load_phaedra_decoder requires model_name=Phaedra_AE_FSQ_4x4")
    return PhaedraDecoderHandle(task=handle.task, device=handle.device)


def decode_phaedra_tokens(handle: PhaedraDecoderHandle, amp_tokens: torch.Tensor, morph_tokens: torch.Tensor) -> torch.Tensor:
    morph_emb = handle.task.model.quantizer.get_codebook_entry(morph_tokens)
    amp_emb = handle.task.model.approximate_continuous.get_codebook_entry(amp_tokens)
    emb = torch.cat([morph_emb, amp_emb], dim=1)
    return handle.task.model.decode(emb)


def decode_tokens(
    handle: TokenDecoderHandle,
    *,
    fsq_tokens: torch.Tensor | None = None,
    amp_tokens: torch.Tensor | None = None,
    morph_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    if handle.model_type == "fsq":
        if fsq_tokens is None:
            raise ValueError("FSQ decode requires fsq_tokens")
        embeds = handle.task.model.quantizer.get_codebook_entry(fsq_tokens)
        return handle.task.model.decode(embeds)

    if amp_tokens is None or morph_tokens is None:
        raise ValueError("Phaedra decode requires amp_tokens and morph_tokens")
    morph_emb = handle.task.model.quantizer.get_codebook_entry(morph_tokens)
    amp_emb = handle.task.model.approximate_continuous.get_codebook_entry(amp_tokens)
    emb = torch.cat([morph_emb, amp_emb], dim=1)
    return handle.task.model.decode(emb)
