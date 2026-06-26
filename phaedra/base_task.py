"""
Base Task Model Interface.

Defines the abstract interface for all Phaedra task models, providing
consistent APIs for training, validation, and tokenized inference.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Mapping, Any, Optional
import torch
import torch.nn as nn

Batch = Mapping[str, torch.Tensor]


class BaseTaskModel(nn.Module, ABC):
    """
    Abstract base class for all research task models.
    Provides consistent interface for:
      - training & validation steps
      - optional tokenized inference pipeline
      - model preparation / optimizer configuration
    """

    # ─── Required interface ────────────────────────────────────────────────
    @abstractmethod
    def configure_optimizers(self) -> list[torch.optim.Optimizer] | tuple[list, list]:
        """Return all optimizers (and optionally schedulers)."""

    # Usually used after accelerator.prepare(model)
    def prepare_model(self, accelerator):
        """Hook for accelerator setup (e.g., EMA wrapping)."""
        return self

    # ─── Core model passes ────────────────────────────────────────────────
    @abstractmethod
    def produce_tokens(self, batch: Batch) -> Any:
        """
        Given raw inputs, produce the model’s discrete / continuous token representation.
        Should return something that `predict_from_tokens` can consume.
        """

    @abstractmethod
    def predict_from_tokens(self, tokens: Any, *args, **kwargs) -> torch.Tensor:
        """
        Given produced tokens, return the model’s prediction (e.g., reconstructed output).
        """

    # ─── High-level task lifecycle ────────────────────────────────────────
    @abstractmethod
    def forward_train(
        self,
        batch: Batch,
        global_step: Optional[int] = None,
        optimizer_idx: int = 0,
    ) -> torch.Tensor:
        """Unified forward pass for training (loss computation)."""

    @abstractmethod
    def forward_val(self, batch: Batch) -> dict[str, float] | None:
        """Unified forward pass for validation (metric computation)."""

    # ─── Loss & metric interface ──────────────────────────────────────────
    @abstractmethod
    def compute_loss(
        self, preds: torch.Tensor, batch: Batch, global_step: Optional[int], optimizer_idx: int
    ) -> torch.Tensor:
        """Compute and return scalar loss used for backprop."""
    
    @abstractmethod
    def compute_metrics(self, preds: torch.Tensor, batch: Batch) -> dict[str, float]:
        """Compute metrics (L1, accuracy, usage stats, etc.)."""

    # ─── Universal forward dispatcher ─────────────────────────────────────
    def forward(
        self,
        batch: Batch,
        mode: str = "train",
        global_step: Optional[int] = None,
        optimizer_idx: int = 0,
        *args,
        **kwargs,
    ):
        """
        Needed for Accelerate/DDP. Dispatches between modes.
        """
        if mode == "train":
            return self.forward_train(batch, optimizer_idx)
        elif mode == "val":
            return self.forward_val(batch)
        elif mode == "produce_tokens":
            return self.produce_tokens(batch)
        elif mode == "predict_from_tokens":
            return self.predict_from_tokens(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")