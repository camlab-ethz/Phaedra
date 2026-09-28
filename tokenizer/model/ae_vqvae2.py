import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import field

# FSQ Tokenizer
from tokenizer.model.building_blocks.quantizers.quantizers import VectorQuantizer2 as VQ2

# VAE Encoder + Decoder
from tokenizer.model.building_blocks.blocks.encoder_decoder import VQ2Decoder, VQ2Encoder


class VQVAE2_AE(nn.Module):
    '''
    Vector Quantized VAE 2 Autoencoder implementation
    '''
    def __init__(self, config):
        super().__init__()

        vae_config = config.vae_hyperparameters
        vq2_config = config.vq2_hyperparameters
        conv_config = config.conv_hyperparameters
       
        self.encoder = VQ2Encoder(
            ch=vae_config.latent_channels, #128
            out_ch=vae_config.input_channels,
            ch_mult=vae_config.encoder_channel_mult,
            num_res_blocks=vae_config.num_res_blocks, # 4
            attn_resolutions=vae_config.attn_resolutions, # [16]
            dropout=vae_config.dropout, # 0.0
            in_channels=vae_config.input_channels,
            resolution=vae_config.input_h, #128
            z_channels=vq2_config.codebook_embed_dim, 
            double_z=vae_config.double_z, # False
        )
        self.decoder = VQ2Decoder(
            ch=vae_config.latent_channels, #128
            out_ch=vae_config.input_channels, #2
            ch_mult=vae_config.decoder_channel_mult, 
            num_res_blocks=vae_config.num_res_blocks, # 4
            attn_resolutions=vae_config.attn_resolutions, # [16]
            dropout=vae_config.dropout, # 0.0
            in_channels=vae_config.input_channels, #2
            resolution=vae_config.input_h, #128
            z_channels=vq2_config.codebook_embed_dim, 
            double_z=vae_config.double_z, # False
        )

        bottleneck_h = int(vae_config.input_h/(2**(len(vae_config.encoder_channel_mult)-1)))
        bottleneck_w = int(vae_config.input_w/(2**(len(vae_config.encoder_channel_mult)-1)))
        HW = (bottleneck_h, bottleneck_w)
        
        # Top and Bottom Quantizers
        self.quantizer_t = VQ2(n_e=vq2_config.codebook_size//4, e_dim=vq2_config.codebook_embed_dim, beta=vq2_config.vq2_beta)
        self.quantizer_b = VQ2(n_e=vq2_config.codebook_size, e_dim=vq2_config.codebook_embed_dim, beta=vq2_config.vq2_beta)
        
        # Convolutions to map encoder channels to codebook dimension
        self.quant_conv_t = nn.Conv2d(vq2_config.codebook_embed_dim, vq2_config.codebook_embed_dim, 1)
        self.quant_conv_b = nn.Conv2d(vq2_config.codebook_embed_dim, vq2_config.codebook_embed_dim, 1)
        
        self.post_quant_conv_t = nn.Conv2d(vq2_config.codebook_embed_dim, vq2_config.codebook_embed_dim, 1)
        self.post_quant_conv_b = nn.Conv2d(vq2_config.codebook_embed_dim, vq2_config.codebook_embed_dim, 1)
        
    def encode(self, x):
        # 1. Hierarchical Encoding
        h_t, h_b = self.encoder(x)
        
        # 2. Quantize Top (Global)
        h_t = self.quant_conv_t(h_t)
        quant_t, diff_t, tokens_t = self.quantizer_t(h_t)
        
        # 3. Quantize Bottom (Local)
        h_b = self.quant_conv_b(h_b)
        quant_b, diff_b, tokens_b = self.quantizer_b(h_b)

        return (quant_t, quant_b), (diff_t + diff_b), (tokens_t, tokens_b), 0.0
 
    def decode(self, quants):
        quant_t, quant_b = quants
        quant_t = self.post_quant_conv_t(quant_t)
        quant_b = self.post_quant_conv_b(quant_b)
        
        return self.decoder(quant_t, quant_b)
    
    def forward(self, x, mode="default"):
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode(x)
        else:
            quants, emb_loss, tokens_hier, usage = self.encode(x)
            recon = self.decode(quants)
            return recon, emb_loss, quants, tokens_hier, usage