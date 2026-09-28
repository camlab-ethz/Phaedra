from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import torch


def _serialize_optimizer(optimizer: Any) -> Any:
    if optimizer is None:
        return None
    if isinstance(optimizer, dict):
        return {key: _serialize_optimizer(value) for key, value in optimizer.items()}
    if isinstance(optimizer, (list, tuple)):
        return [_serialize_optimizer(value) for value in optimizer]
    return optimizer.state_dict()


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    cfg: dict,
    extra_state: dict[str, Any] | None = None,
) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": _serialize_optimizer(optimizer),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "config": cfg,
    }
    if extra_state:
        payload.update(extra_state)

    torch.save(
        payload,
        ckpt_path,
    )


def load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    ckpt_path = Path(path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    from hub.utils.weights import is_released, load_weights
    if is_released(ckpt_path):
        # released weights: model state only (no optimizer) -> usable as --pretrained, not for --resume
        state, info = load_weights(ckpt_path, map_location=device)
        return {"model": state, "epoch": info["epoch"], "step": info["step"], "released": True}
    if ckpt_path.is_dir():
        from hub.utils.weights import resolve_weights_file
        ckpt_path = resolve_weights_file(ckpt_path)
    try:
        payload = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError:
        # PyTorch >=2.6 defaults to weights_only=True; our training checkpoints
        # include optimizer/RNG state, so they require full pickle loading.
        print(
            "[checkpoint] torch.load(weights_only=True) failed; retrying with weights_only=False "
            f"for trusted checkpoint: {ckpt_path}"
        )
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format at {ckpt_path}")
    if "model" not in payload:
        raise ValueError(f"Checkpoint missing 'model' state: {ckpt_path}")
    return payload
