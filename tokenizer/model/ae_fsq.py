import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import field

# FSQ Tokenizer
from tokenizer.model.building_blocks.quantizers.fsq_quant import FSQ

# VAE Encoder + Decoder
from tokenizer.model.building_blocks.blocks.encoder_decoder import Decoder, Encoder


class FSQ_AE(nn.Module):
    '''
    This model uses the standard AE architecture with a VAR+FSQ tokenizer and an additional (approximately) continuous channel for high‑fidelity reconstruction.
    '''
    def __init__(self, config):
        super().__init__()

        vae_config = config.vae_hyperparameters
        fsq_config = config.fsq_hyperparameters
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
            z_channels=fsq_config.codebook_embed_dim, 
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
            z_channels=fsq_config.codebook_embed_dim, 
            double_z=vae_config.double_z, # False
        )

        bottleneck_h = int(vae_config.input_h/(2**(len(vae_config.encoder_channel_mult)-1)))
        bottleneck_w = int(vae_config.input_w/(2**(len(vae_config.encoder_channel_mult)-1)))
        HW = (bottleneck_h, bottleneck_w)
        
        levels = fsq_config.levels
        self.quantizer =  FSQ(levels=levels, dim=fsq_config.codebook_embed_dim, scale=fsq_config.fsq_scale)
        
        self.quant_conv = torch.nn.Conv2d(fsq_config.codebook_embed_dim, fsq_config.codebook_embed_dim, conv_config.quant_conv_ks, stride=1, padding=conv_config.quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(fsq_config.codebook_embed_dim, fsq_config.codebook_embed_dim, conv_config.quant_conv_ks, stride=1, padding=conv_config.quant_conv_ks//2)
    
    def encode(self, x):
        encodings = self.encoder(x)
        encodings = self.quant_conv(encodings)
        quant, tokens_hier = self.quantizer(encodings)

        loss_codebook = F.mse_loss(quant, encodings.detach())
        loss_commitment = F.mse_loss(encodings, quant.detach())
        beta = 0.25
        emb_loss = loss_codebook + beta * loss_commitment


        return quant, emb_loss, tokens_hier, 0.0
 
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, x, mode="default"):
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode(x)
        else:
            quant, emb_loss, tokens_hier, usage = self.encode(x)
            dec = self.decode(quant)
            return dec, emb_loss, quant, tokens_hier, usage