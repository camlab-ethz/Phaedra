
import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from argparse import ArgumentParser

Tensor = torch.Tensor


def permute_tensor(tensor: Tensor, kernel_dim: int) -> Tensor:
    if kernel_dim == 1:
        # Reshape for the 1D case
        return tensor.permute(0, 2, 1)
    elif kernel_dim == 2:
        # Reshape for the 2D case
        return tensor.permute(0, 3, 2, 1)
    elif kernel_dim == 3:
        # Reshape for the 3D case
        return tensor.permute(0, 4, 3, 2, 1)
    else:
        raise ValueError(
            f"Unsupported kernel_dim={kernel_dim}. Only 1D, 2D, and 3D data are valid."
        )


def reshape_jax_torch(tensor: Tensor, kernel_dim: int = None) -> Tensor:
    """
    A jax based dataloader is off shape (bs, width, height, depth, c),
    while a PyTorch based dataloader is off shape (bs, c, depth, height, width).

    It transforms a tensor for the 2D and 3D case as follows:
    - 2D: (bs, c, depth, height, width) <-> (bs, width, height, depth, c)
    - 3D: (bs, c, height, width) <-> (bs, width, height, c)

    Code can be used either dynamics or static.
    - dynamic: if kernel_dim is None
    - static: if kernel_dim
    """
    if kernel_dim is None:
        # Infer kernel_dim dynamically based on tensor.ndim
        kernel_dim = tensor.ndim - 2  # Extract batch_size and channel

    return permute_tensor(tensor, kernel_dim)


def default_init(scale: float = 1e-10):
    """Initialization of weights and biases with scaling"""

    def initializer(tensor: Tensor):
        """We need to differentiate between biases and weights"""

        if tensor.ndim == 1:  # if bias
            bound = torch.sqrt(torch.tensor(3.0)) * scale
            with torch.no_grad():
                return tensor.uniform_(-bound, bound)

        else:  # if weights
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            std = torch.sqrt(torch.tensor(scale / ((fan_in + fan_out) / 2.0)))
            bound = torch.sqrt(torch.tensor(3.0)) * std  # uniform dist. scaling factor
            with torch.no_grad():
                return tensor.uniform_(-bound, bound)

    return initializer

