"""
Phaedra System - Training and Inference Wrapper.

This module provides the high-level system wrapper that handles:
- Optimizer and scheduler configuration
- Training and validation loops  
- EMA updates
- Tokenized encode → decode inference pipeline
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_ema import ExponentialMovingAverage

from .base_task import BaseTaskModel, Batch
from .edemamix import AdEMAMix
from .phaedra_model import PhaedraModel as Phaedra_AE
from .utils import denormalize_tensor, compute_token_usage


class PhaedraSystem(BaseTaskModel):
    """
    High-level system wrapper for the Phaedra autoencoder.
    Handles optimizer configuration, training/validation, EMA updates,
    and tokenized inference (encode → decode from tokens).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Phaedra_AE(cfg.tokenizer_hyperparameters)
        self.l1 = nn.L1Loss()
        self.ema = None

    # ──────────────────────────────────────────────────────────────────────
    # Optimizer / scheduler configuration
    # ──────────────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        hp = self.cfg.training_hyperparameters
        opt_g = AdEMAMix(
            self.model.parameters(),
            lr=hp.lr,
            betas=(hp.beta1, hp.beta2, hp.beta3),
            alpha=hp.alpha,
            beta3_warmup=hp.beta3_warmup,
            alpha_warmup=hp.alpha_warmup,
            weight_decay=hp.weight_decay,
        )
        
        scheduler = ReduceLROnPlateau(
            opt_g,
            mode="min",
            factor=hp.lr_factor,
            patience=hp.lr_patience,
            min_lr=hp.lr_min,
            verbose=True,
        )
        return [opt_g], [scheduler]

    # ──────────────────────────────────────────────────────────────────────
    # Model preparation (Accelerate / EMA)
    # ──────────────────────────────────────────────────────────────────────
    def prepare_model(self, accelerator):
        self.model = accelerator.prepare(self.model)
        self.module = self.model.module if hasattr(self.model, "module") else self.model
        self.ema = ExponentialMovingAverage(
            self.model.parameters(),
            decay=self.cfg.training_hyperparameters.ema,
        )
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Encoding / decoding interface
    # ──────────────────────────────────────────────────────────────────────
    def produce_tokens(self, batch: Batch):
        """Encodes input batch into hierarchical discrete + continuous tokens."""
        x = batch["field_variables_in"]
        _, _, tokens_hierarchy, _ = self.model(x, mode="encode")
        return tokens_hierarchy

    def predict_from_tokens(self, token_hierarchy):
        """Decodes full reconstruction from hierarchical token set."""
        morph_tokens, amp_tokens = token_hierarchy
    

        # decode hierarchical + continuous embeddings
        morph_embeddings = self.module.quantizer.get_codebook_entry(morph_tokens)
        amp_embeddings = self.module.approximate_continuous.get_codebook_entry(amp_tokens)

        # full latent + decode
        full_embeddings = torch.cat([morph_embeddings, amp_embeddings], dim=1)
        outputs = self.model(full_embeddings, mode="decode")
        return outputs

    # Convenience helper for experiments
    def reconstruct_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end encode → decode for inference/evaluation."""
        tokens = self.model(x, mode="encode")[2]
        return self.predict_from_tokens(tokens)

    # ──────────────────────────────────────────────────────────────────────
    # Training / validation
    # ──────────────────────────────────────────────────────────────────────
    def forward_train(self, batch: Batch, optimizer_idx: int):
        """Standard training step (reconstruction + quantization losses)."""
        x = batch["field_variables_in"]
        recon, emb_loss, *_ = self.model(x)
        loss_recon = self.l1(recon, x)
        return loss_recon + emb_loss

    def forward_val(self, batch: Batch):
        """Validation step returning (reconstruction, L1 loss, usage %)."""
        with self.ema.average_parameters():
            return self._shared_eval(batch)

    def _shared_eval(self, batch: Batch):
        """Shared evaluation logic for validation and inference verification."""

        x = batch["field_variables_in"]
        recon, emb_loss, quant, tokens_hier, usage = self.model(x)

        # Compute average usage percentage
        usage_avg = 0.0
        for tokens_set in tokens_hier[0]: # tokens_hier[0] is a list of the VAR tokens
            usage_pct, _, _ = compute_token_usage(tokens_set[-1])
            usage_avg += usage_pct
        usage_pct = usage_avg / len(tokens_hier[0])

        # Denormalize reconstruction
        mean = batch["field_variables_in_mean"]
        std = batch["field_variables_in_std"]
        recon = denormalize_tensor(recon, mean, std)

        # Compute L1 loss against ground truth
        true_values = batch["field_variables_out"]
        l1_loss = self.l1(recon, true_values)

        # Return tuple (used in your trainer)
        return recon, l1_loss.item(), usage_pct

    def verify_inference(self, batch: Batch):
        """Sanity check: encode → decode matches direct forward pass."""
        x = batch["field_variables_in"]
        with torch.no_grad():
            # Direct forward pass
            recon_direct, _, quant, tokens_1, *_ = self.model(x)

            # Encode → decode pass
            # Check token consistency
            tokens_2 = self.model(x, mode="encode")[2]
            for t1, t2 in zip(tokens_1[0], tokens_2[0]):
                for level in range(len(t1)):
                    assert torch.allclose(t1[level], t2[level], atol=1e-7), f"Inference verification failed!"
            assert torch.allclose(tokens_1[1], tokens_2[1], atol=1e-7), "Inference verification failed -- Continuous Token Discrepancies!"

            # Check embedding reconstructions from tokens
            amp_embeddings_1 = quant[:, -1:, :, :]
            amp_embeddings_2 = self.model.approximate_continuous.get_codebook_entry(tokens_2[1])
            assert torch.allclose(amp_embeddings_1, amp_embeddings_2, atol=1e-7), "Inference verification failed -- Continuous Embedding Discrepancies!"
            
            morph_embeddings_1 = quant[:, :-1, :, :]
            morph_embeddings_2 = self.model.quantizer.get_codebook_entry(tokens_2[0])
            assert torch.allclose(morph_embeddings_1, morph_embeddings_2, atol=1e-6), "Inference verification failed -- Morph Embedding Discrepancies!"
            
            out_1 = self.model.decoder(self.model.post_quant_conv(torch.cat([morph_embeddings_1, amp_embeddings_1], dim=1)))  # Warm-up pass for any lazy modules
            out_2 = self.model.decoder(self.model.post_quant_conv(torch.cat([morph_embeddings_2, amp_embeddings_2], dim=1)))
            assert torch.allclose(out_1, out_2, atol=1e-2), "Inference verification failed -- Decoded Output Discrepancies!"

            recon_encode_decode = self.predict_from_tokens(tokens_2)
            assert torch.allclose(recon_direct, recon_encode_decode, atol=1e-2), "Inference verification failed!"

    # ──────────────────────────────────────────────────────────────────────
    # BaseTaskModel hooks
    # ──────────────────────────────────────────────────────────────────────
    def compute_loss(self, preds, batch, global_step, optimizer_idx):
        """Loss used for backprop (L1 + quantization)."""
        x = batch["field_variables_in"]
        if isinstance(preds, tuple):
            recon, quant_loss, *_ = preds
        elif isinstance(preds, dict):
            recon = preds["recon"]
            quant_loss = preds.get("quant_loss", 0.0)
        else:
            recon, quant_loss = preds, 0.0
        return self.l1(recon, x) + quant_loss

    def compute_metrics(self, preds, batch):
        """Compute validation metrics for logging."""
        if isinstance(preds, tuple):
            recon = preds[0]
        elif isinstance(preds, dict):
            recon = preds["recon"]
        else:
            recon = preds

        true = batch["field_variables_out"]
        l1_loss = self.l1(recon, true).item()
        usage = preds[2] if isinstance(preds, tuple) and len(preds) > 2 else 0.0
        return {"l1_loss": l1_loss, "usage": usage}