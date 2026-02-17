"""
Test script to verify Phaedra installation.

Run this script from the repository root as a module:
    python -m phaedra.test_model

This will:
1. Create a model with default configuration
2. Run a forward pass with random data
3. Verify encode/decode consistency
"""

import sys
import torch
from omegaconf import OmegaConf
from pathlib import Path


def test_phaedra_model():
    """Test basic model functionality."""
    print("Testing Phaedra model...")
    
    from phaedra.phaedra_model import PhaedraModel

    # Load config (relative to this package directory)
    package_dir = Path(__file__).resolve().parent
    config = OmegaConf.load(package_dir / "model_config.yaml")
    model_config = config.tokenizer_hyperparameters
    
    # Create model
    model = PhaedraModel(model_config)
    model.eval()
    
    # Test forward pass
    batch_size = 2
    h, w = model_config.vae_hyperparameters.input_h, model_config.vae_hyperparameters.input_w
    c = model_config.vae_hyperparameters.input_channels
    
    x = torch.randn(batch_size, c, h, w)
    
    with torch.no_grad():
        # Full forward pass
        recon, emb_loss, quant, tokens, usage = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Reconstruction shape: {recon.shape}")
        print(f"  Quantized latent shape: {quant.shape}")
        print(f"  Embedding loss: {emb_loss.item():.4f}")
        
        # Test encode/decode consistency
        quant_enc, _, tokens_enc, _ = model.encode(x)
        recon_dec = model.decode(quant_enc)
        
        assert torch.allclose(quant, quant_enc, atol=1e-6), "Encode output mismatch!"
        assert torch.allclose(recon, recon_dec, atol=1e-6), "Decode output mismatch!"
        
    print("  ✓ Model forward pass successful!")
    print("  ✓ Encode/decode consistency verified!")
    return True


def test_fsq_quantizer():
    """Test FSQ quantizer."""
    print("\nTesting FSQ quantizer...")
    
    from phaedra.fsq_quant import FSQ
    
    # Create quantizer
    levels = [5, 4, 4, 3, 3, 3, 2, 2]
    quantizer = FSQ(levels=levels, dim=8, scale=10.0)
    
    # Test quantization
    x = torch.randn(2, 8, 16, 16)
    
    with torch.no_grad():
        quant, indices = quantizer(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Quantized shape: {quant.shape}")
        print(f"  Indices shape: {indices.shape}")
        print(f"  Codebook size: {quantizer.codebook_size}")
        
        # Test codebook lookup
        reconstructed = quantizer.get_codebook_entry(indices)
        assert torch.allclose(quant, reconstructed, atol=1e-6), "Codebook lookup mismatch!"
        
    print("  ✓ FSQ quantization successful!")
    print("  ✓ Codebook lookup verified!")
    return True


def test_continuous_layer():
    """Test continuous tokenization layer."""
    print("\nTesting continuous tokenization layer...")
    
    from phaedra.phaedra_layer import ContinuousTokenizerLayer
    from omegaconf import OmegaConf
    
    # Create config
    ct_config = OmegaConf.create({
        "quantizer": "FSQ",
        "continuous_L": 1024,
        "continuous_scale": 0.1,
    })
    
    # Create layer
    layer = ContinuousTokenizerLayer(ct_config)
    
    # Test forward pass
    x = torch.randn(2, 1, 16, 16)
    
    with torch.no_grad():
        quant, loss, tokens = layer(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Quantized shape: {quant.shape}")
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Commitment loss: {loss.item():.4f}")
        
        # Test codebook entry
        reconstructed = layer.get_codebook_entry(tokens)
        assert torch.allclose(quant, reconstructed, atol=1e-6), "Codebook entry mismatch!"
        
    print("  ✓ Continuous tokenization successful!")
    print("  ✓ Codebook entry verified!")
    return True


def test_encoder_decoder():
    """Test encoder and decoder."""
    print("\nTesting encoder/decoder...")
    
    from phaedra.encoder_decoder import Encoder, Decoder
    
    # Config
    ch = 64
    in_channels = 1
    resolution = 128
    z_channels = 9
    ch_mult = [2, 2, 4]
    
    # Create encoder
    encoder = Encoder(
        ch=ch,
        out_ch=in_channels,
        ch_mult=ch_mult,
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        in_channels=in_channels,
        resolution=resolution,
        z_channels=z_channels,
        double_z=False,
    )
    
    # Create decoder
    decoder = Decoder(
        ch=ch,
        out_ch=in_channels,
        ch_mult=ch_mult,
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        in_channels=in_channels,
        resolution=resolution,
        z_channels=z_channels,
    )
    
    # Test forward pass
    x = torch.randn(2, in_channels, resolution, resolution)
    
    with torch.no_grad():
        z = encoder(x)
        recon = decoder(z)
        
        expected_z_h = resolution // (2 ** (len(ch_mult) - 1))
        
        print(f"  Input shape: {x.shape}")
        print(f"  Latent shape: {z.shape}")
        print(f"  Reconstruction shape: {recon.shape}")
        
        assert z.shape == (2, z_channels, expected_z_h, expected_z_h), f"Unexpected latent shape!"
        assert recon.shape == x.shape, "Reconstruction shape mismatch!"
        
    print("  ✓ Encoder forward pass successful!")
    print("  ✓ Decoder forward pass successful!")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Phaedra Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    try:
        all_passed &= test_fsq_quantizer()
        all_passed &= test_continuous_layer()
        all_passed &= test_encoder_decoder()
        all_passed &= test_phaedra_model()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! Installation is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
