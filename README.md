# Phaedra: Learning High-Fidelity Discrete Tokenization for the Physical Sciences

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **Phaedra**, a tokenization framework for high fidelity reconstructions.

## Overview

Phaedra introduces a hybrid tokenization approach for data that separates:
- **Morphological features** — Structural patterns quantized with Finite Scalar Quantization (FSQ)
- **Amplitude features** — Continuous value distributions captured with approximate continuous quantization

This separation enables efficient discrete representations while preserving the precise magnitudes of physical measurements.

<p align="center">
  <img src="docs/phaedra_pipeline.pdf" width="700" alt="Phaedra Pipeline">
</p>

## Installation

```bash
# Clone the repository
git clone https://github.com/camlab-ethz/Phaedra.git
cd Phaedra

# Install with pip
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
import torch
from phaedra import PhaedraModel, PhaedraSystem
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("model_config.yaml")

# Initialize model
model = PhaedraModel(config.tokenizer_hyperparameters)

# Encode input to tokens
x = torch.randn(1, 1, 128, 128)  # [B, C, H, W]
quant, emb_loss, tokens, usage = model.encode(x)

# Decode back to reconstruction
reconstruction = model.decode(quant)
```

### Training

```bash
# Train with default configuration
python -m phaedra.train --config model_config.yaml

# Train with custom settings
python -m phaedra.train \
    --config model_config.yaml \
    --experiment_name my_experiment \
    --epochs 100 \
    --batch_size 16
```

### Inference

```python
from phaedra import PhaedraSystem
from omegaconf import OmegaConf

# Load trained model
config = OmegaConf.load("model_config.yaml")
system = PhaedraSystem(config)
system.load_state_dict(torch.load("checkpoint.pt"))
system.eval()

# Tokenize and reconstruct
with torch.no_grad():
    tokens = system.produce_tokens(batch)
    reconstruction = system.predict_from_tokens(tokens)
```

## Model Architecture

### Components

| Component | Description |
|-----------|-------------|
| **Encoder** | Convolutional encoder with ResNet blocks and self-attention |
| **FSQ Quantizer** | Finite Scalar Quantization for morphological tokens |
| **Continuous Layer** | High-resolution FSQ for amplitude quantization |
| **Decoder** | Symmetric decoder with upsampling and attention |

### Configuration

Key hyperparameters in `model_config.yaml`:

```yaml
tokenizer_hyperparameters:
  vae_hyperparameters:
    input_channels: 1          # Input channels
    encoder_channel_mult: [2, 2, 4]  # Downsampling factors
    latent_channels: 128       # Base channel count
    
  fsq_hyperparameters:
    fsq_L: [5, 4, 4, 3, 3, 3, 2, 2]  # FSQ levels per dimension
    codebook_embed_dim: 9      # Embedding dimension
    fsq_scale: 10.0            # FSQ scaling factor
    
  ct_hyperparameters:
    continuous_L: 1024         # Continuous quantization levels
    continuous_scale: 0.1      # Continuous scaling factor
```

## Data Format

The dataloader should return batches with the following structure:

```python
batch = {
    "field_variables_in": tensor,      # Input [B, C, H, W]
    "field_variables_out": tensor,     # Target [B, C, H, W]  
    "field_variables_in_mean": tensor, # Per-sample mean
    "field_variables_in_std": tensor,  # Per-sample std
}
```

See `train.py` for the `create_dataloader()` interface.

## Training Details

- **Optimizer**: AdEMAMix (adaptive EMA mixing)
- **Learning Rate**: 1e-4 with ReduceLROnPlateau scheduler
- **Mixed Precision**: BF16 training via HuggingFace Accelerate
- **EMA**: Exponential moving average with decay 0.999

### Distributed Training

```bash
# Multi-GPU training with Accelerate
accelerate launch --num_processes 4 -m phaedra.train --config model_config.yaml
```

## Project Structure

```
phaedra/
├── __init__.py           # Package exports
├── phaedra_model.py      # Core autoencoder model
├── phaedra_layer.py      # Continuous tokenization layer
├── task_phaedra.py       # Training system wrapper
├── base_task.py          # Abstract task interface
├── encoder_decoder.py    # Encoder/decoder architectures
├── fsq_quant.py          # Finite Scalar Quantization
├── edemamix.py           # AdEMAMix optimizer
├── train.py              # Training entry point
├── train_loop.py         # Training loop implementation
├── utils.py              # Utility functions
├── model_config.yaml     # Default configuration
└── README.md
```

## Citation

If you use Phaedra in your research, please cite:

```bibtex
@misc{lingsch2026phaedra,
      title={Phaedra: Learning High-Fidelity Discrete Tokenization for the Physical Science}, 
      author={Levi Lingsch and Georgios Kissas and Johannes Jakubik and Siddhartha Mishra},
      year={2026},
      eprint={2602.03915},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03915}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FSQ implementation based on [Mentzer et al., 2024](https://arxiv.org/abs/2309.15505)
- AdEMAMix optimizer from [Pagliardini et al., 2024](https://arxiv.org/abs/2409.03137)
