import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_ema import ExponentialMovingAverage

from tokenizer.core import BaseTaskModel, Batch
from tokenizer.core.edemamix import AdEMAMix
from tokenizer.model.ae_vqvae2 import VQVAE2_AE
from tokenizer.utils.normalization import denormalize_tensor
from tokenizer.utils.token_usage import compute_token_usage

class VQVAE2AESystem(BaseTaskModel):
    """
    High-level system wrapper for the VQ-VAE-2 autoencoder.
    Handles hierarchical latent spaces (Top/Bottom), EMA, and dual-quantizer logic.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Note: using vqvae2 hyperparameters from config
        self.model = VQVAE2_AE(cfg.vqvae2_hyperparameters)
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

        scheduler = CosineAnnealingLR(
            opt_g,
            T_max=max(1, int(getattr(hp, "cosine_t_max", getattr(hp, "lr_patience", hp.epochs)))),
            eta_min=getattr(hp, "cosine_eta_min", getattr(hp, "lr_min", 0.0)),
        )
        return [opt_g], [scheduler]

    # ──────────────────────────────────────────────────────────────────────
    # Model preparation (Accelerate / EMA)
    # ──────────────────────────────────────────────────────────────────────
    def prepare_model(self, accelerator):
        self.model = accelerator.prepare(self.model)
        self.ema = ExponentialMovingAverage(
            self.model.parameters(),
            decay=self.cfg.training_hyperparameters.ema,
        )
        return self

    # ──────────────────────────────────────────────────────────────────────
    # Encoding / decoding interface
    # ──────────────────────────────────────────────────────────────────────
    def produce_tokens(self, batch: Batch):
        """Encodes input into a tuple of (tokens_top, tokens_bottom)."""
        x = batch["field_variables_in"]
        _, _, tokens_hierarchy, _ = self.model(x, mode="encode")
        return tokens_hierarchy # (id_t, id_b)

    def predict_from_tokens(self, tokens):
        """Reconstructs the field from a tuple of hierarchical tokens."""
        id_t, id_b = tokens

        # Get embeddings from respective codebooks
        quant_t = self.model.quantizer_t.get_codebook_entry(id_t)
        quant_b = self.model.quantizer_b.get_codebook_entry(id_b)

        # Decode using the hierarchical latent pair
        outputs = self.model((quant_t, quant_b), mode="decode")
        return outputs

    def reconstruct_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end inference helper."""
        tokens = self.model(x, mode="encode")[2]
        return self.predict_from_tokens(tokens)

    # ──────────────────────────────────────────────────────────────────────
    # Training / validation
    # ──────────────────────────────────────────────────────────────────────
    def forward_train(self, batch: Batch, optimizer_idx: int):
        """Hierarchical training: Loss = Recon + Sum(Quantization Losses)."""
        x = batch["field_variables_in"]
        recon, total_emb_loss, *_ = self.model(x)
        loss_recon = self.l1(recon, x)
        return loss_recon + total_emb_loss

    def forward_val(self, batch: Batch):
        with self.ema.average_parameters():
            return self._shared_eval(batch)

    def _shared_eval(self, batch: Batch):
        x = batch["field_variables_in"]
        recon, emb_loss, quants, tokens, usage = self.model(x)

        # tokens is (id_t, id_b). Compute usage for both and average.
        usage_t, _, _ = compute_token_usage(tokens[0])
        usage_b, _, _ = compute_token_usage(tokens[1])
        avg_usage = (usage_t + usage_b) / 2

        # Denormalize for metric calculation
        mean = batch["field_variables_in_mean"]
        std = batch["field_variables_in_std"]
        recon_denorm = denormalize_tensor(recon, mean, std)

        true_values = batch["field_variables_out"]
        l1_loss = self.l1(recon_denorm, true_values)

        return recon_denorm, l1_loss.item(), avg_usage

    def verify_inference(self, batch: Batch):
        """Checks consistency across hierarchical levels."""
        x = batch["field_variables_in"]
        with torch.no_grad():
            # 1. Direct forward pass
            recon_direct, _, (q_t1, q_b1), (t_t1, t_b1), *_ = self.model(x)

            # 2. Encode -> tokens pass
            _, _, (t_t2, t_b2), _ = self.model(x, mode="encode")

            # Check Top and Bottom Token consistency
            assert torch.allclose(t_t1, t_t2), "Top level token mismatch!"
            assert torch.allclose(t_b1, t_b2), "Bottom level token mismatch!"

            # 3. Predict from tokens (Inference path)
            recon_from_tokens = self.predict_from_tokens((t_t2, t_b2))
            
            assert torch.allclose(recon_direct, recon_from_tokens, atol=1e-2), \
                "Inference verification failed: Reconstruction discrepancy!"

    # ──────────────────────────────────────────────────────────────────────
    # BaseTaskModel hooks
    # ──────────────────────────────────────────────────────────────────────
    def compute_loss(self, preds, batch, global_step, optimizer_idx):
        x = batch["field_variables_in"]
        if isinstance(preds, tuple):
            # Assumes model forward returns (recon, emb_loss, ...)
            recon, quant_loss = preds[0], preds[1]
        else:
            recon, quant_loss = preds, 0.0
        return self.l1(recon, x) + quant_loss

    def compute_metrics(self, preds, batch):
        if isinstance(preds, tuple):
            recon = preds[0]
            usage = preds[2] if len(preds) > 2 else 0.0
        else:
            recon, usage = preds, 0.0

        true = batch["field_variables_out"]
        l1_loss = self.l1(recon, true).item()
        return {"l1_loss": l1_loss, "usage": usage}