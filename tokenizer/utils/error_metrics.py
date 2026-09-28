import numpy as np
import torch
import torch.nn.functional as F


def _flatten_metric_tensors(pred: torch.Tensor, target: torch.Tensor):
    """Flattens tensors so each sample/channel pair becomes one row."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if pred.ndim < 2:
        raise ValueError("Expected tensors with at least batch and channel dimensions")

    pred_flat = pred.reshape(-1, int(np.prod(pred.shape[2:])))
    target_flat = target.reshape(-1, int(np.prod(target.shape[2:])))
    return pred_flat, target_flat


def _safe_rel_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-12):
    return numerator / torch.clamp(denominator, min=eps)


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Returns gradient magnitude over the last two spatial dimensions."""
    if x.ndim < 4:
        raise ValueError("Expected input of shape [B, C, H, W] (or compatible)")

    dy, dx = torch.gradient(x, dim=(-2, -1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def rel_l1_error_tensor(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Relative L1 error in percent, aggregated across the full tensor."""
    num = torch.mean(torch.abs(pred - target))
    den = torch.mean(torch.abs(target))
    return _safe_rel_ratio(num, den, eps=eps) * 100


def rel_l2_error_tensor(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Relative L2 error in percent, aggregated across the full tensor."""
    num = torch.sqrt(torch.mean((pred - target) ** 2))
    den = torch.sqrt(torch.mean(target ** 2))
    return _safe_rel_ratio(num, den, eps=eps) * 100


def top_percent_abs_rel_l1_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    top_percent: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Relative L1 error on the top-k% region defined by |target|."""
    if not (0.0 < top_percent <= 100.0):
        raise ValueError(f"top_percent must be in (0, 100], got {top_percent}")

    pred_flat, target_flat = _flatten_metric_tensors(pred, target)
    abs_target = torch.abs(target_flat)

    q = 1.0 - (top_percent / 100.0)
    thresholds = torch.quantile(abs_target, q=q, dim=1, keepdim=True)
    mask = abs_target >= thresholds

    abs_error = torch.abs(pred_flat - target_flat)
    num = torch.sum(abs_error * mask, dim=1)
    den = torch.sum(abs_target * mask, dim=1)
    rel = _safe_rel_ratio(num, den, eps=eps) * 100
    return rel.mean()


def gradient_top_percent_abs_rel_l1_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    top_percent: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Relative L1 error on the top-k% of gradient magnitude |grad(target)|."""
    pred_grad = gradient_magnitude(pred)
    target_grad = gradient_magnitude(target)
    return top_percent_abs_rel_l1_error(pred_grad, target_grad, top_percent=top_percent, eps=eps)


def tail_conditional_rel_l1_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    tail_quantile: float = 0.95,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Relative L1 error conditioned on |target| above a quantile threshold."""
    if not (0.0 < tail_quantile < 1.0):
        raise ValueError(f"tail_quantile must be in (0, 1), got {tail_quantile}")

    pred_flat, target_flat = _flatten_metric_tensors(pred, target)
    abs_target = torch.abs(target_flat)
    threshold = torch.quantile(abs_target, q=tail_quantile, dim=1, keepdim=True)
    mask = abs_target >= threshold

    num = torch.sum(torch.abs(pred_flat - target_flat) * mask, dim=1)
    den = torch.sum(abs_target * mask, dim=1)
    rel = _safe_rel_ratio(num, den, eps=eps) * 100
    return rel.mean()


def tail_conditional_rel_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    tail_quantile: float = 0.95,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Relative L2 error conditioned on |target| above a quantile threshold."""
    if not (0.0 < tail_quantile < 1.0):
        raise ValueError(f"tail_quantile must be in (0, 1), got {tail_quantile}")

    pred_flat, target_flat = _flatten_metric_tensors(pred, target)
    abs_target = torch.abs(target_flat)
    threshold = torch.quantile(abs_target, q=tail_quantile, dim=1, keepdim=True)
    mask = abs_target >= threshold

    sq_error = (pred_flat - target_flat) ** 2
    sq_target = target_flat ** 2
    num = torch.sqrt(torch.sum(sq_error * mask, dim=1))
    den = torch.sqrt(torch.sum(sq_target * mask, dim=1))
    rel = _safe_rel_ratio(num, den, eps=eps) * 100
    return rel.mean()

def norm_l1_error(orig, rec, scale):
    """Computes the relative L1 error between original and reconstructed fields."""
    l1_error = torch.mean(torch.abs(orig - rec))
    scale_tensor = torch.as_tensor(scale, device=l1_error.device, dtype=l1_error.dtype)
    scale_value = torch.clamp(torch.mean(torch.abs(scale_tensor)), min=1e-12)
    rel_error = (l1_error / scale_value) * 100  # percentage
    return rel_error.item()

def norm_l2_error(orig, rec, scale):
    """Computes the relative L2 error between original and reconstructed fields."""
    l2_error = torch.mean((orig - rec) ** 2).sqrt()
    scale_tensor = torch.as_tensor(scale, device=l2_error.device, dtype=l2_error.dtype)
    scale_value = torch.clamp(torch.mean(torch.abs(scale_tensor)), min=1e-12)
    rel_error = (l2_error / scale_value) * 100  # percentage
    return rel_error.item()

def rel_l1_error(orig, rec):
    """Computes the relative L1 error between original and reconstructed fields."""
    return rel_l1_error_tensor(rec, orig).item()

def rel_l2_error(orig, rec):
    """Computes the relative L2 error between original and reconstructed fields."""
    return rel_l2_error_tensor(rec, orig).item()

def get_radial_spectrum_torch(field):
    """Computes the 1D radial power spectrum of a 2D field using PyTorch."""
    # Ensure field is at least 2D; handles (H, W)
    device = field.device
    nx, ny = field.shape[-2:]
    
    # 1. Compute 2D FFT and shift low frequencies to center
    fft_val = torch.fft.fftshift(torch.fft.fft2(field, norm="forward"))
    psd2D = torch.abs(fft_val)**2
    
    # 2. Create coordinate grid
    # indexing='ij' gives us the same behavior as np.indices
    y = torch.linspace(-(ny // 2), ny // 2 - 1 + (ny % 2), ny, device=device)
    x = torch.linspace(-(nx // 2), nx // 2 - 1 + (nx % 2), nx, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # 3. Calculate radial distances (integer bins)
    r = torch.sqrt(grid_x**2 + grid_y**2).to(torch.long)
    
    # 4. Sum power in radial bins
    # flatten both to 1D; psd2D acts as weights for the binning
    tbin = torch.bincount(r.reshape(-1), weights=psd2D.reshape(-1))
    
    return tbin

def tail_fidelity(orig_field, rec_field, tail_threshold=1.0):
    """
    Calculates energy fidelity specifically for the high-frequency 'tail'.
    tail_threshold: 0.5 means the upper 50% of the wavenumbers.
    """
    p_orig = get_radial_spectrum_torch(orig_field)
    p_rec = get_radial_spectrum_torch(rec_field)
    
    # Slice the arrays to only look at high wavenumbers
    cutoff = int(len(p_orig) * (1 - tail_threshold))
    p_orig_tail = p_orig[cutoff:]
    p_rec_tail = p_rec[cutoff:]
    
    recovered = torch.sum(torch.minimum(p_orig_tail, p_rec_tail))
    total = torch.sum(p_orig_tail)
    
    return (recovered / (total + 1e-6)).item() * 100

def log_tail_fidelity(orig_field, rec_field, tail_threshold=1.0, floor_db=-6):
    """
    Calculates the percentage of the log-spectral area recovered in the tail.
    tail_threshold: 0.5 means the upper 50% of wavenumbers.
    floor_db: How many orders of magnitude below peak to consider 'relevant'.
    """
    p_orig = get_radial_spectrum_torch(orig_field)
    p_rec = get_radial_spectrum_torch(rec_field)
    
    # 1. Define the noise floor (epsilon)
    epsilon = p_orig.max() * (10**floor_db)
    
    # 2. Isolate the tail
    cutoff = int(len(p_orig) * (1 - tail_threshold))
    p_orig_tail = p_orig[cutoff:]
    p_rec_tail = p_rec[cutoff:]
    
    # 3. Transform to Log-Space (Log10)
    log_orig = torch.log10(p_orig_tail + epsilon)
    log_rec = torch.log10(p_rec_tail + epsilon)
    log_floor = torch.log10(epsilon.clone().detach()) 
    
    # 4. Calculate 'Log-Area' Error
    # Total available log-area above the floor
    total_log_area = torch.sum(torch.abs(log_orig - log_floor))
    # Unrecovered log-area (the distance between curves)
    unrecovered_log_area = torch.sum(torch.abs(log_orig - log_rec))
    
    # 5. Fidelity percentage
    fidelity = (1 - (unrecovered_log_area / (total_log_area + 1e-6))) * 100
    
    return fidelity.clamp(0, 100).item()

def spectral_energy_fidelity(orig, rec):
    """Calculates SEF percentage entirely on device."""
    # 1. Get radial power spectra
    p_orig = get_radial_spectrum_torch(orig)
    p_rec = get_radial_spectrum_torch(rec)
    
    # 2. Align lengths (radial distances can vary slightly based on grid)
    min_len = min(len(p_orig), len(p_rec))
    p_orig = p_orig[:min_len]
    p_rec = p_rec[:min_len]
    
    # 3. Calculate Fidelity
    # Using torch.minimum handles noise injection (aliasing)
    recovered_energy = torch.sum(torch.minimum(p_orig, p_rec))
    total_energy = torch.sum(p_orig)
    
    # Return as a float (percentage)
    return (recovered_energy / total_energy).item() * 100

def standardized_max_error(orig, rec, global_sigma):
    """
    Calculates the maximum point-wise error normalized by global sigma.
    Captures the single 'worst' artifact in the spatial domain.
    """
    max_err = torch.max(torch.abs(orig - rec))
    sigma_tensor = torch.as_tensor(global_sigma, device=max_err.device, dtype=max_err.dtype)
    sigma_value = torch.clamp(torch.mean(torch.abs(sigma_tensor)), min=1e-12)
    return (max_err / sigma_value).item() * 100

def max_spectral_difference(orig, rec, floor_db=-12):
    """
    Finds the maximum discrepancy in the 2D power spectrum.
    Useful for identifying periodic artifacts/spurs.
    """
    # 2D Power Spectra
    p_orig = torch.abs(torch.fft.fft2(orig))**2
    p_rec = torch.abs(torch.fft.fft2(rec))**2
    
    # Floor to avoid machine precision noise
    epsilon = p_orig.max() * (10**floor_db)
    
    # Log-space difference
    log_diff = torch.abs(torch.log10(p_orig + epsilon) - torch.log10(p_rec + epsilon))
    
    return torch.max(log_diff).item()


def local_variance_error(orig, rec, window_size=7):
    """
    Computes the maximum difference in local variance between orig and rec.
    Identifies localized blurring or blocky artifacts.
    """
    def get_local_var(img, w):
        # Padding to keep dimensions consistent
        padding = w // 2
        
        # E[X^2]
        mu_sq = F.avg_pool2d(img**2, w, stride=1, padding=0)
        # (E[X])^2
        sq_mu = F.avg_pool2d(img, w, stride=1, padding=0)**2
        
        return (mu_sq - sq_mu).squeeze()

    var_orig = get_local_var(orig, window_size)
    var_rec = get_local_var(rec, window_size)

    # Normalize by global variance to get relative differences
    # Return the maximum discrepancy in local 'busyness'
    return (torch.max(torch.abs(var_orig - var_rec))/torch.max(var_orig)).item() * 100

def min_spectral_coherence(orig, rec, smoothing_kernel=7, floor_db=-4):
    """
    Calculates Magnitude Squared Coherence and returns the MINIMUM
    only for regions with significant signal power.
    """
    # 1. Fourier transforms
    f_orig = torch.fft.fft2(orig)
    f_rec = torch.fft.fft2(rec)
    
    # 2. Densities
    P_oo = torch.abs(f_orig)**2
    P_rr = torch.abs(f_rec)**2
    P_or = f_orig * torch.conj(f_rec)
    
    # 3. Smoothing
    padding = smoothing_kernel // 2
    
    def smooth_real(x):
        return F.avg_pool2d(x, kernel_size=smoothing_kernel, stride=1, padding=padding)

    def smooth_complex(x):
        # Using cat(dim=1) to handle [1, 1, H, W] inputs correctly
        x_stack = torch.cat([x.real, x.imag], dim=1) 
        smoothed = F.avg_pool2d(x_stack, kernel_size=smoothing_kernel, stride=1, padding=padding)
        return torch.complex(smoothed[:, 0:1, :, :], smoothed[:, 1:2, :, :])

    S_oo = smooth_real(P_oo)
    S_rr = smooth_real(P_rr)
    S_or = smooth_complex(P_or)
    
    # 4. Magnitude Squared Coherence
    coherence = (torch.abs(S_or)**2) / (S_oo * S_rr + 1e-15)
    
    # Define a threshold based on peak power (e.g., -10 orders of magnitude down)
    peak_power = torch.max(S_oo)
    threshold = peak_power * (10**floor_db)
    
    # Only look at pixels where the original signal is above the noise floor
    significant_mask = S_oo > threshold
    
    if not torch.any(significant_mask):
        return 0.0
        
    # Take the minimum only where the signal actually exists
    relevant_coherence = coherence[significant_mask]
    return torch.min(torch.clamp(relevant_coherence, 0, 1)).item()

class GlobalTokenAnalyst:
    def __init__(self, vocab_size, device='cpu'):
        self.vocab_size = vocab_size
        # Track frequency of every token index
        self.counts = torch.zeros(vocab_size, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, token_ids):
        # --- DEBUGGING START ---
        # print(f"DEBUG: Received type {type(token_ids).__name__}")
        # --- DEBUGGING END ---

        # 1. Handle VAR Hierarchy Objects
        if type(token_ids).__name__ == 'VarTokensHierarchyBatch':
            # Check standard VAR attributes
            token_ids = token_ids[0]
            token_ids = list(token_ids)

        # 2. Handle Lists (Hierarchical) and tuples (Phaedra)
        if isinstance(token_ids, (list)):
            if len(token_ids) == 0:
                return
            # Use only the highest resolution (last level)
            return self.update(token_ids[-1])
        elif isinstance(token_ids, tuple):
            if len(token_ids) == 0:
                return
            return self.update(token_ids[0])

        # 3. Standard Tensor Processing
        if not isinstance(token_ids, torch.Tensor):
            return

        # Ensure tokens are integers and on the same device as our counter
        flat_tokens = token_ids.detach().flatten().to(self.counts.device).long()
        
        # Check the range of tokens we actually found
        # If this prints [0, 0] or something weird, we know the source is empty
        # print(f"DEBUG: Token Range [{flat_tokens.min()}, {flat_tokens.max()}] | Vocab Size: {self.vocab_size}")

        # 4. Range Masking (The most likely culprit for 0.00 metrics)
        # If tokens are outside [0, vocab_size-1], they are ignored.
        valid_mask = (flat_tokens >= 0) & (flat_tokens < self.vocab_size)
        valid_tokens = flat_tokens[valid_mask]
        
        if valid_tokens.numel() > 0:
            # minlength ensures the resulting tensor matches vocab_size
            batch_counts = torch.bincount(valid_tokens, minlength=self.vocab_size)
            self.counts += batch_counts

    def compute(self):
        total_tokens = self.counts.sum().item()
        
        # 1. Utilization
        num_used = (self.counts > 0).sum().item()
        util_pct = (num_used / self.vocab_size) * 100 if self.vocab_size > 0 else 0

        if total_tokens == 0:
            return {
                "vocab_size": self.vocab_size, # Added this key
                "utilization_pct": util_pct, 
                "unique_used": num_used,
                "entropy_bits": 0, 
                "redundancy_pct": 0
            }

        # 2. Entropy (H)
        probs = self.counts[self.counts > 0] / total_tokens
        entropy = -torch.sum(probs * torch.log2(probs)).item()

        # 3. Redundancy (R)
        max_entropy = np.log2(self.vocab_size)
        redundancy = (1 - (entropy / max_entropy)) * 100 if max_entropy > 0 else 0

        return {
            "vocab_size": self.vocab_size,     # Fix: Added this key
            "utilization_pct": util_pct,
            "unique_used": num_used,
            "entropy_bits": entropy,
            "redundancy_pct": redundancy,
            "total_tokens_processed": total_tokens
        }