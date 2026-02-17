"""
Phaedra Training Entry Point.

Main entry point for training Phaedra models with configurable datasets
and model configurations.

Usage:
    python -m phaedra.train --config configs/model_config.yaml
"""

import torch
from omegaconf import OmegaConf
import argparse
from pathlib import Path

from .train_loop import fit
from .task_phaedra import PhaedraSystem


def main(args):
    """Main training entry point."""
    # Load configuration
    config_path = Path(args.config) if args.config else Path(__file__).parent / "model_config.yaml"
    model_config = OmegaConf.load(config_path)

    # Override config with command line arguments if provided
    if args.experiment_name:
        model_config.training_hyperparameters.experiment_name = args.experiment_name
    if args.epochs:
        model_config.training_hyperparameters.epochs = args.epochs
    if args.batch_size:
        model_config.system_config.dataloader_kwargs.batch_size = args.batch_size

    # Create model
    task = PhaedraSystem(model_config)
    
    # Create dataloaders (implement your own data loading here)
    train_dataloader = create_dataloader(model_config, split="train")
    val_dataloader = create_dataloader(model_config, split="val")

    print(f"Training Phaedra model: {model_config.training_hyperparameters.experiment_name}")
    print(f"  - Epochs: {model_config.training_hyperparameters.epochs}")
    print(f"  - Batch size: {model_config.system_config.dataloader_kwargs.batch_size}")
    
    fit(task, model_config, train_loader=train_dataloader, val_loader=val_dataloader)


def create_dataloader(config, split="train"):
    """Create a dataloader for the specified split.
    
    This is a placeholder - implement your own data loading logic here.
    The dataloader should return batches with the following keys:
        - 'field_variables_in': Input tensor [B, C, H, W]
        - 'field_variables_out': Target tensor [B, C, H, W] (for validation)
        - 'field_variables_in_mean': Per-sample mean for denormalization
        - 'field_variables_in_std': Per-sample std for denormalization
    
    Args:
        config: OmegaConf configuration object
        split: One of 'train', 'val', or 'test'
    
    Returns:
        DataLoader instance
    """
    raise NotImplementedError(
        "Implement create_dataloader() with your dataset. "
        "See docstring for expected batch format."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phaedra tokenizer model")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to YAML config file (default: model_config.yaml)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Override experiment name")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    args = parser.parse_args()

    main(args)
