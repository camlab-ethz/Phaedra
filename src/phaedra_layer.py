"""
Continuous Tokenizer Layer for Phaedra.

Implements the approximate continuous tokenization scheme using FSQ
for high-resolution amplitude quantization.
"""

import torch.nn.functional as F
import torch.nn as nn

from .fsq_quant import FSQ

class ContinuousTokenizerLayer(nn.Module):
    """Continuous tokenizer layer using high-resolution FSQ.
    
    Quantizes amplitude information using a single-dimension FSQ with
    a large number of levels for near-continuous representation.
    
    Args:
        config: Configuration object with:
            - quantizer: Quantizer type (currently only "FSQ")
            - continuous_L: Number of quantization levels (e.g., 1024)
            - continuous_scale: Scale factor for FSQ
    """
    
    def __init__(self, config):
        super().__init__()
        self.quantizer_name = config.quantizer
        if self.quantizer_name == "FSQ":
            self.quantizer = FSQ(
                levels=[config.continuous_L], 
                dim=1, 
                scale=config.continuous_scale
            )
        else:
            raise NotImplementedError(f"Quantizer {self.quantizer_name} not implemented in ContinuousTokenizerLayer.")
    
    def forward(self, x):
        """Forward pass through quantizer.
        
        Args:
            x: Input tensor [B, 1, H, W]
            
        Returns:
            Tuple of (quantized tensor, commitment loss, token indices)
        """
        if self.quantizer_name == "FSQ":
            quantized, tokens = self.quantizer(x)
            diff = F.mse_loss(quantized, x)
        return quantized, diff, tokens
    
    def get_codebook_entry(self, indices):
        """Decode token indices back to embeddings.
        
        Args:
            indices: Token indices tensor
            
        Returns:
            Decoded embeddings tensor
        """
        return self.quantizer.get_codebook_entry(indices)