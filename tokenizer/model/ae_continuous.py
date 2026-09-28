import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import field

# VAE Encoder + Decoder
from tokenizer.model.building_blocks.blocks.encoder_decoder import Decoder, Encoder


class Continuous_AE(nn.Module):
    '''
    This model uses the standard AE architecture with no quantization as an ablation.
    '''
    def __init__(self, config):
        super().__init__()

        vae_config = config.vae_hyperparameters
        var_config = config.var_hyperparameters
        conv_config = config.conv_hyperparameters
       
        self.encoder = Encoder(
            ch=vae_config.latent_channels, #128
            out_ch=vae_config.input_channels,
            ch_mult=vae_config.encoder_channel_mult,
            num_res_blocks=vae_config.num_res_blocks, # 4
            attn_resolutions=vae_config.attn_resolutions, # [16]
            dropout=vae_config.dropout, # 0.0
            in_channels=vae_config.input_channels,
            resolution=vae_config.input_h, #128
            z_channels=var_config.codebook_embed_dim, 
            double_z=vae_config.double_z, # False
        )
        self.decoder = Decoder(
            ch=vae_config.latent_channels, #128
            out_ch=vae_config.input_channels, #2
            ch_mult=vae_config.decoder_channel_mult, 
            num_res_blocks=vae_config.num_res_blocks, # 4
            attn_resolutions=vae_config.attn_resolutions, # [16]
            dropout=vae_config.dropout, # 0.0
            in_channels=vae_config.input_channels, #2
            resolution=vae_config.input_h, #128
            z_channels=var_config.codebook_embed_dim, 
            double_z=vae_config.double_z, # False
        )

        
        self.quant_conv = torch.nn.Conv2d(var_config.codebook_embed_dim, var_config.codebook_embed_dim, conv_config.quant_conv_ks, stride=1, padding=conv_config.quant_conv_ks//2)
        
        self.post_quant_conv = torch.nn.Conv2d(var_config.codebook_embed_dim, var_config.codebook_embed_dim, conv_config.quant_conv_ks, stride=1, padding=conv_config.quant_conv_ks//2)

    def encode(self, x):
        encodings = self.encoder(x)
        encodings = self.quant_conv(encodings)
        return encodings
 
    def decode(self, encodings):
        encodings = self.post_quant_conv(encodings)
        dec = self.decoder(encodings)
        return dec
    
    def forward(self, x, mode="default"):
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode(x)
        else:
            encodings = self.encode(x)
            dec = self.decode(encodings)
            return dec