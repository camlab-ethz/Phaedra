"""Uniform weight I/O for every model in this repository.

Two on-disk formats are accepted wherever a model is loaded:

* **Released weights** (Hugging Face): ``model.safetensors`` (with ``config.yaml``
  and ``metadata.json`` next to it). These are exactly the weights the paper
  evaluated -- EMA weights wherever the paper used EMA -- so no EMA file is
  applied on top ("ema_baked").
* **Training outputs of this code base**: ``checkpoint_last.pt`` /
  ``checkpoint_step_*.pt`` / ``checkpoint_epoch_*.pt`` (a dict with ``"model"``
  and optionally ``"ema"``, ``"epoch"``, ``"step"``, ``"config"``) or the
  tokenizer's accelerate layout ``pytorch_model.bin`` (+ ``ema.pt``).

A directory argument resolves to the first existing file in
``WEIGHT_FILE_PRIORITY``; a file argument is used as-is.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

WEIGHT_FILE_PRIORITY = ("model.safetensors", "checkpoint_last.pt", "pytorch_model.bin")
_PREFIXES = ("module.", "model.", "_orig_mod.")


def resolve_weights_file(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_dir():
        for name in WEIGHT_FILE_PRIORITY:
            if (p / name).is_file():
                return p / name
        for pattern in ("*.safetensors", "*.pt", "*.pth", "*.bin"):
            matches = sorted(q for q in p.glob(pattern) if q.name != "ema.pt")
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No weight file in {p} (looked for {', '.join(WEIGHT_FILE_PRIORITY)}, *.pt, *.bin)")
    if not p.is_file():
        raise FileNotFoundError(f"Weight file not found: {p}")
    return p


def is_released(path: str | Path) -> bool:
    """True if `path` resolves to released (safetensors, EMA-baked) weights."""
    try:
        return resolve_weights_file(path).suffix == ".safetensors"
    except FileNotFoundError:
        return False


def normalize_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip a wrapper prefix (DDP `module.`, `model.`, torch.compile `_orig_mod.`)
    when *every* key carries it."""
    keys = list(state_dict)
    for prefix in _PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            keys = list(state_dict)
    return state_dict


def _extract_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
        return payload
    raise ValueError(f"Unsupported checkpoint payload type: {type(payload)}")


def load_weights(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    """Load model weights from a released file / training checkpoint / directory.

    Returns ``(state_dict, info)`` where ``info`` has
      file          resolved file path
      released      True for safetensors (EMA already applied where applicable)
      metadata      safetensors string metadata ({} for torch files)
      payload       full torch payload (dict) or None for safetensors
      epoch, step   ints when known (payload or metadata), else None
    """
    f = resolve_weights_file(path)
    if f.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        device = str(map_location) if not isinstance(map_location, torch.device) else str(map_location)
        state = load_file(str(f), device=device)
        with safe_open(str(f), framework="pt") as fh:
            meta = dict(fh.metadata() or {})
        info = {"file": f, "released": True, "metadata": meta, "payload": None,
                "epoch": _int_or_none(meta.get("epoch")), "step": _int_or_none(meta.get("step"))}
        return state, info
    payload = torch.load(str(f), map_location=map_location, weights_only=False)
    state = normalize_keys(_extract_state(payload))
    info = {"file": f, "released": False, "metadata": {}, "payload": payload if isinstance(payload, dict) else None,
            "epoch": _int_or_none(payload.get("epoch")) if isinstance(payload, dict) else None,
            "step": _int_or_none(payload.get("step")) if isinstance(payload, dict) else None}
    return state, info


def load_into(model: torch.nn.Module, path: str | Path, map_location: str | torch.device = "cpu",
              strict: bool | None = None) -> dict[str, Any]:
    """Load weights into `model`. Released weights are loaded strictly by default
    (they were written from this code's own `state_dict()`); training checkpoints
    non-strictly, but a missing key is always an error. Returns `info` (see load_weights)."""
    state, info = load_weights(path, map_location=map_location)
    if strict is None:
        strict = bool(info["released"])
    res = model.load_state_dict(state, strict=strict)
    if not strict and res.missing_keys:
        raise RuntimeError(f"{info['file']}: {len(res.missing_keys)} missing keys, e.g. {res.missing_keys[:5]}")
    info["unexpected_keys"] = list(getattr(res, "unexpected_keys", []) or [])
    return info


def save_released(state_dict: dict[str, torch.Tensor], path: str | Path, metadata: dict[str, Any] | None = None) -> str:
    """Write `state_dict` as safetensors (fp32/int tensors kept as-is, cloned so no
    storage is shared) with string metadata. Returns the file's sha256."""
    from safetensors.torch import save_file

    tensors = {k: v.detach().to("cpu").clone().contiguous() for k, v in state_dict.items()}
    meta = {str(k): (v if isinstance(v, str) else json.dumps(v)) for k, v in (metadata or {}).items()}
    meta.setdefault("format", "pt")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    return sha256_file(path)


def sha256_file(path: str | Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _int_or_none(x: Any) -> int | None:
    try:
        return None if x is None else int(x)
    except (TypeError, ValueError):
        return None
