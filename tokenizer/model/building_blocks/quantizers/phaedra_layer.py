# Approximate Continuous Tokenizer 
import torch.nn.functional as F
import torch.nn as nn
from tokenizer.model.building_blocks.quantizers.fsq_quant import FSQ
from tokenizer.model.building_blocks.quantizers.quantizers import IndexPropagationQuantize

class ContinuousTokenizerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantizer_name = config.quantizer
        if self.quantizer_name == "IBQ":
            self.quantizer = IndexPropagationQuantize(
                levels=config.continuous_L, 
                dim=1, 
                beta=config.quantizer_commit_loss_beta,
                use_entropy_loss=config.use_entropy_loss
            )
        elif self.quantizer_name == "FSQ":
            self.quantizer = FSQ(
                levels=[config.continuous_L], 
                dim=1, 
                scale=config.continuous_scale
            )
    def forward(self, x):
        if self.quantizer_name == "FSQ":
            quantized, tokens = self.quantizer(x)
            diff = F.mse_loss(quantized, x)

            # loss_codebook = F.mse_loss(quantized, x.detach())
            # loss_commitment = F.mse_loss(x, quantized.detach())
            # beta = 0.25
            # diff = loss_codebook + beta * loss_commitment

        elif self.quantizer_name == "IBQ":
            quantized, diff, tokens = self.quantizer(x)
            
        return quantized, diff, tokens