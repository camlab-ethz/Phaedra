import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_ema import ExponentialMovingAverage

from tokenizer.core import BaseTaskModel, Batch
from tokenizer.core.edemamix import AdEMAMix
from tokenizer.model.ae_continuous import Continuous_AE
from tokenizer.utils.normalization import denormalize_tensor
from tokenizer.utils.token_usage import compute_token_usage


class ContinuousAESystem(BaseTaskModel):
    """
    High-level system wrapper for the Phaedra autoencoder.
    Handles optimizer configuration, training/validation, EMA updates,
    and tokenized inference (encode → decode from tokens).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = Continuous_AE(cfg.var_tokenizer_hyperparameters)
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
        """Encodes input batch into hierarchical discrete + continuous tokens."""
        x = batch["field_variables_in"]
        encodings = self.model(x, mode="encode")
        return encodings

    def predict_from_tokens(self, encodings):
        """Decodes full reconstruction from hierarchical token set."""

        # full latent + decode
        outputs = self.model(encodings, mode="decode")
        return outputs

    # Convenience helper for experiments
    def reconstruct_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """End-to-end encode → decode for inference/evaluation."""
        encodings = self.model(x, mode="encode")
        return self.predict_from_tokens(encodings)

    # ──────────────────────────────────────────────────────────────────────
    # Training / validation
    # ──────────────────────────────────────────────────────────────────────
    def forward_train(self, batch: Batch, optimizer_idx: int):
        """Standard training step (reconstruction + quantization losses)."""
        x = batch["field_variables_in"]
        recon = self.model(x)
        loss_recon = self.l1(recon, x)
        return loss_recon

    def forward_val(self, batch: Batch):
        """Validation step returning (reconstruction, L1 loss, usage %)."""
        with self.ema.average_parameters():
            return self._shared_eval(batch)

    def _shared_eval(self, batch: Batch):
        """Shared evaluation logic for validation and inference verification."""

        x = batch["field_variables_in"]
        recon = self.model(x)

        # Denormalize reconstruction
        mean = batch["field_variables_in_mean"]
        std = batch["field_variables_in_std"]
        recon = denormalize_tensor(recon, mean, std)

        # Compute L1 loss against ground truth
        true_values = batch["field_variables_out"]
        l1_loss = self.l1(recon, true_values)

        # Return tuple (used in your trainer)
        return recon, l1_loss.item(), 0.0

    def verify_inference(self, batch: Batch):
        """Sanity check: encode → decode matches direct forward pass."""
        x = batch["field_variables_in"]
        with torch.no_grad():
            # Direct forward pass
            recon_direct = self.model(x)

            # Encode → decode pass
            # Check token consistency
            encodings = self.model(x, mode="encode")
            recon_encode_decode = self.predict_from_tokens(encodings)
            assert torch.allclose(recon_direct, recon_encode_decode, atol=1e-5), "Inference verification failed!"

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