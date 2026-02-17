"""
Utility Functions for Phaedra.

Contains helper functions for:
- Tensor normalization/denormalization
- Token usage statistics
- Logging and checkpointing
- Visualization utilities
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Normalization Utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_tensor(x: torch.Tensor, dim: Tuple[int, ...] = (-2, -1)) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize tensor to zero mean and unit variance.
    
    Args:
        x: Input tensor
        dim: Dimensions over which to compute statistics
    
    Returns:
        Tuple of (normalized tensor, mean, std)
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True) + 1e-8
    return (x - mean) / std, mean, std


def denormalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor using provided mean and std.
    
    Args:
        x: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor
    """
    return x * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Token Usage Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_usage(tokens: torch.Tensor) -> Tuple[float, int, int]:
    """Compute token usage statistics.
    
    Args:
        tokens: Token indices tensor
    
    Returns:
        Tuple of (usage percentage, unique count, total possible)
    """
    unique_tokens = torch.unique(tokens)
    n_unique = len(unique_tokens)
    # Estimate codebook size from max token index (conservative)
    codebook_size = tokens.max().item() + 1
    usage_pct = 100.0 * n_unique / codebook_size
    return usage_pct, n_unique, codebook_size


# ─────────────────────────────────────────────────────────────────────────────
# Logging Utilities
# ─────────────────────────────────────────────────────────────────────────────

def create_logger(
    accelerator,
    logging_dir: str = "logs",
    enable_wandb: bool = False,
    wandb_init_kwargs: Optional[dict] = None,
) -> logging.Logger:
    """Create a logger with optional Weights & Biases integration.
    
    Args:
        accelerator: HuggingFace Accelerator instance
        logging_dir: Directory for log files
        enable_wandb: Whether to enable W&B logging
        wandb_init_kwargs: Keyword arguments for wandb.init()
    
    Returns:
        Configured logger instance
    """
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"{logging_dir}/training.log"),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
        
        if enable_wandb:
            try:
                import wandb
                wandb_init_kwargs = wandb_init_kwargs or {}
                wandb.init(**wandb_init_kwargs)
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    
    return logger


def log_metric(
    logger: logging.Logger,
    name: str,
    value: float,
    epoch: int,
    step: int,
    log_wandb: bool = False,
) -> None:
    """Log a metric to logger and optionally to W&B.
    
    Args:
        logger: Logger instance
        name: Metric name
        value: Metric value
        epoch: Current epoch
        step: Current global step
        log_wandb: Whether to also log to W&B
    """
    logger.info(f"[Epoch {epoch}, Step {step}] {name}: {value:.4f}")
    
    if log_wandb:
        try:
            import wandb
            wandb.log({name: value, "epoch": epoch, "step": step})
        except ImportError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    accelerator,
    global_step: int,
    checkpoint_dir: str = "checkpoints",
    name: str = "checkpoint",
    ema: Optional[object] = None,
) -> str:
    """Save a training checkpoint.
    
    Args:
        accelerator: HuggingFace Accelerator instance
        global_step: Current global training step
        checkpoint_dir: Directory to save checkpoints
        name: Checkpoint name
        ema: Optional EMA wrapper to save
    
    Returns:
        Path to saved checkpoint
    """
    if accelerator.is_main_process:
        save_path = Path(checkpoint_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)
        
        accelerator.save_state(str(save_path))
        
        if ema is not None:
            torch.save(ema.state_dict(), save_path / "ema.pt")
        
        # Save metadata
        torch.save({"global_step": global_step}, save_path / "metadata.pt")
        
        return str(save_path)
    return ""


def load_checkpoint(
    accelerator,
    checkpoint_dir: str,
    name: str,
    ema: Optional[object] = None,
) -> int:
    """Load a training checkpoint.
    
    Args:
        accelerator: HuggingFace Accelerator instance
        checkpoint_dir: Directory containing checkpoints
        name: Checkpoint name to load
        ema: Optional EMA wrapper to restore
    
    Returns:
        Global step at checkpoint
    """
    load_path = Path(checkpoint_dir) / name
    
    if not load_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    
    accelerator.load_state(str(load_path))
    
    if ema is not None and (load_path / "ema.pt").exists():
        ema.load_state_dict(torch.load(load_path / "ema.pt"))
    
    metadata = torch.load(load_path / "metadata.pt")
    return metadata.get("global_step", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_contourf_comparison(
    x_true: torch.Tensor,
    x_recon: torch.Tensor,
    save_path: str,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    """Save a comparison figure of ground truth vs reconstruction.
    
    Args:
        x_true: Ground truth tensor [B, C, H, W]
        x_recon: Reconstructed tensor [B, C, H, W]
        save_path: Path to save the figure
        num_samples: Number of samples to display
        figsize: Figure size
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to numpy
    x_true = x_true.detach().cpu().numpy()
    x_recon = x_recon.detach().cpu().numpy()
    
    num_samples = min(num_samples, x_true.shape[0])
    n_channels = x_true.shape[1]
    
    fig, axes = plt.subplots(num_samples, n_channels * 2, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        for c in range(n_channels):
            # Ground truth
            ax_true = axes[i, c * 2]
            im_true = ax_true.contourf(x_true[i, c], levels=50, cmap='viridis')
            ax_true.set_title(f"GT Ch{c}" if i == 0 else "")
            ax_true.axis('off')
            
            # Reconstruction
            ax_recon = axes[i, c * 2 + 1]
            im_recon = ax_recon.contourf(x_recon[i, c], levels=50, cmap='viridis')
            ax_recon.set_title(f"Recon Ch{c}" if i == 0 else "")
            ax_recon.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_token_histogram(
    tokens: torch.Tensor,
    save_path: str,
    codebook_size: Optional[int] = None,
) -> None:
    """Plot histogram of token usage.
    
    Args:
        tokens: Token indices tensor
        save_path: Path to save the figure
        codebook_size: Optional codebook size for x-axis limit
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    tokens_np = tokens.detach().cpu().numpy().flatten()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(tokens_np, bins=min(100, len(np.unique(tokens_np))), edgecolor='black')
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Token Usage Distribution")
    
    if codebook_size is not None:
        ax.set_xlim(0, codebook_size)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
