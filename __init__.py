"""Phaedra Tokenizer.

A tokenization framework that combines:
- Finite Scalar Quantization (FSQ) for morphological features
- Approximate continuous quantization for amplitude features

This enables efficient representation of geospatial data while preserving
both structural patterns and continuous value distributions.
"""

from .phaedra_model import PhaedraModel
from .phaedra_layer import ContinuousTokenizerLayer
from .task_phaedra import PhaedraSystem
from .base_task import BaseTaskModel
from .fsq_quant import FSQ
from .encoder_decoder import Encoder, Decoder

__version__ = "1.0.0"
__author__ = "Phaedra Team"

__all__ = [
    "PhaedraModel",
    "PhaedraSystem", 
    "ContinuousTokenizerLayer",
    "BaseTaskModel",
    "FSQ",
    "Encoder",
    "Decoder",
]
