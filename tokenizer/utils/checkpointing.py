from pathlib import Path
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import math

def save_checkpoint(
    accelerator: Accelerator,
    step: int,
    checkpoint_dir: str | Path = "./checkpoints",
    name: str = "checkpoint",
    ema=None,
):
    """
    Save model, optimizer, and EMA state using Accelerate.
    """
    accelerator.print(f"Saving checkpoint {step}.")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # save accelerator state
    accelerator.save_state(checkpoint_dir / name, safe_serialization=False)

    # save EMA state if available (only on main process)
    if ema is not None and accelerator.is_main_process:
        torch.save(ema.state_dict(), checkpoint_dir / name /"ema.pt")


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str | Path = "./checkpoints",
    name: str = "checkpoint",
    ema=None,
):
    """
    Load model, optimizer, and EMA state using Accelerate.
    """
    ema_loaded = False

    accelerator.print(f"Loading checkpoint {name}.")
    checkpoint_dir = Path(checkpoint_dir)

    # load model/optimizer/scheduler state written by accelerator.save_state
    # (the research code had this line commented out, so `load_from_checkpoint`
    # only restored the EMA weights; a real resume needs the full state)
    accelerator.load_state(checkpoint_dir / name)

    # load EMA if available
    ema_path = checkpoint_dir / name / "ema.pt"
    if ema is not None and ema_path.exists():
        ema.load_state_dict(torch.load(ema_path, map_location=accelerator.device))
        ema_loaded = True

    assert ema_loaded, "ema state is not loaded"


# def interpolate_positional_embeddings(state_dict, model_state):
#     """
#     Interpolate positional embeddings in `state_dict` to match `model_state` shapes.

#     Handles the following common cases:
#     - pretrained: [L, C] or [1, L, C] (e.g. ViT with optional cls token) → target [L_new, C]
#     - pretrained: [C, H, W] or [1, C, H, W] → target [C, H_new, W_new] or [1, C, H_new, W_new]
#     - row_emb/col_emb: [C, H] or [C, W] → target [C, H_new] or [C, W_new]
#     - preserves leading special tokens (e.g. class token) if present
#     """
#     updated = False

#     def is_square(n):
#         s = int(math.sqrt(n))
#         return s * s == n

#     for k in list(state_dict.keys()):
#         # Handle row_emb and col_emb separately (2D separable positional embeddings)
#         if "row_emb" in k or "col_emb" in k:
#             old = state_dict[k]
#             if k not in model_state:
#                 continue
#             new = model_state[k]
            
#             if old.shape == new.shape:
#                 continue
            
#             # Expected shape: [C, spatial_dim]
#             if old.ndim == 2 and new.ndim == 2:
#                 C_old, spatial_old = old.shape
#                 C_new, spatial_new = new.shape
                
#                 if C_old != C_new:
#                     print(f"[pos_emb skip] {k}: channel dimension mismatch {C_old} != {C_new}")
#                     continue
                
#                 # Interpolate along the spatial dimension
#                 # Reshape to [1, C, spatial_old] for interpolation
#                 old_reshaped = old.unsqueeze(0)  # [1, C, spatial_old]
#                 resized = F.interpolate(old_reshaped, size=spatial_new, mode="linear", align_corners=False)
#                 resized = resized.squeeze(0)  # [C, spatial_new]
                
#                 state_dict[k] = resized
#                 print(f"[pos_emb resize] {k}: {tuple(old.shape)} → {tuple(resized.shape)}")
#                 updated = True
#             else:
#                 print(f"[pos_emb skip] {k}: unexpected shape {old.shape} -> {new.shape}")
#             continue
        
#         if "pos_emb" in k or "pos_embed" in k or "pos_emb_layers" in k:
#             old = state_dict[k]
#             if k not in model_state:
#                 continue
#             new = model_state[k]

#             if old.shape == new.shape:
#                 continue

#             # Case A: channel-first spatial tensor [C, H, W] or [1, C, H, W]
#             if (old.ndim == 3 or old.ndim == 4) and (new.ndim == 3 or new.ndim == 4):
#                 # normalize to [1, C, H, W]
#                 old_t = old.unsqueeze(0) if old.ndim == 3 else old
#                 new_t = new.unsqueeze(0) if new.ndim == 3 else new
#                 _, C_old, H_old, W_old = old_t.shape
#                 _, C_new, H_new, W_new = new_t.shape
#                 if (H_old, W_old) != (H_new, W_new):
#                     resized = F.interpolate(old_t, size=(H_new, W_new), mode="bilinear", align_corners=False)
#                     resized = resized.squeeze(0) if old.ndim == 3 else resized
#                     state_dict[k] = resized
#                     print(f"[pos_emb resize] {k}: {tuple(old.shape)} → {tuple(resized.shape)}")
#                     updated = True
#                 else:
#                     # shapes differ in channels or batch dim — fall back to non-interpolating assignment
#                     state_dict[k] = old
#                 continue

#             # Case B: sequence-style embeddings [L, C] or [1, L, C]
#             if old.ndim == 2 or (old.ndim == 3 and old.shape[0] == 1):
#                 # normalize to [L, C]
#                 old_seq = old.squeeze(0) if old.ndim == 3 else old
#                 new_seq = new.squeeze(0) if new.ndim == 3 else new

#                 L_old, C_old = old_seq.shape
#                 L_new, C_new = new_seq.shape

#                 # detect number of special tokens at start (e.g. cls token)
#                 num_special = 0
#                 if not is_square(L_old) and is_square(L_old - 1):
#                     num_special = 1
#                 elif not is_square(L_old) and is_square(L_old - 2):
#                     # rare case: two special tokens
#                     num_special = 2

#                 L_grid_old = L_old - num_special
#                 L_grid_new = L_new - num_special if L_new > num_special else L_new

#                 # require grid sizes to be perfect squares
#                 if L_grid_old > 0 and is_square(L_grid_old) and is_square(L_grid_new):
#                     H_old = W_old = int(math.sqrt(L_grid_old))
#                     H_new = W_new = int(math.sqrt(L_grid_new))

#                     # extract grid part and reshape to [1, C, H, W]
#                     grid_old = old_seq[num_special:].transpose(0, 1).reshape(1, C_old, H_old, W_old)
#                     # interpolate spatial grid
#                     grid_resized = F.interpolate(grid_old, size=(H_new, W_new), mode="bilinear", align_corners=False)
#                     # reshape back to [L_new - num_special, C]
#                     grid_resized_flat = grid_resized.reshape(C_old, H_new * W_new).transpose(0, 1)

#                     if num_special:
#                         special_tokens = old_seq[:num_special]
#                         resized = torch.cat([special_tokens, grid_resized_flat], dim=0)
#                     else:
#                         resized = grid_resized_flat

#                     state_dict[k] = resized
#                     print(f"[pos_emb resize] {k}: {tuple(old.shape)} → {tuple(resized.shape)} (special={num_special})")
#                     updated = True
#                 else:
#                     # fallback: try 1D linear interpolation along length (preserve channels)
#                     old_1d = old_seq.transpose(0, 1).unsqueeze(0)  # [1, C, L]
#                     try:
#                         resized_1d = F.interpolate(old_1d, size=L_new, mode="linear", align_corners=False).squeeze(0).transpose(0, 1)
#                         state_dict[k] = resized_1d
#                         print(f"[pos_emb 1D resize] {k}: {tuple(old.shape)} → {tuple(resized_1d.shape)}")
#                         updated = True
#                     except Exception:
#                         # leave as-is if interpolation fails
#                         print(f"[pos_emb skip] {k}: cannot interpolate from {old.shape} to {new.shape}")
#                 continue

#             # Unknown layout — skip
#             print(f"[pos_emb skip] {k}: unsupported shape {old.shape} -> {new.shape}")

#     return state_dict, updated



# def load_checkpoint(
#     accelerator: Accelerator,
#     checkpoint_dir: str | Path = "./checkpoints",
#     name: str = "checkpoint",
#     model=None,
#     ema=None,
# ):
#     """
#     Load model, optimizer, and EMA state using Accelerate.
#     Includes interpolation of positional embeddings for resolution changes.
#     """
#     accelerator.print(f"Loading checkpoint {name}.")
#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoint_path = checkpoint_dir / name
#     # If a model is provided, pre-process the saved model state dict on disk to
#     # interpolate positional embeddings so that accelerator.load_state doesn't
#     # fail due to size mismatches.
#     state_dict_path = checkpoint_path / "pytorch_model.bin"
#     if model is not None and state_dict_path.exists():
#         try:
#             sd = torch.load(state_dict_path, map_location="cpu")
#             sd, updated = interpolate_positional_embeddings(sd, model.state_dict())
#             if updated:
#                 torch.save(sd, state_dict_path)
#                 accelerator.print("✅ Preprocessed checkpoint: resized positional embeddings.")
#         except Exception as e:
#             accelerator.print(f"⚠️ Preprocessing checkpoint failed: {e}. Continuing to load normally.")

#     # --- Load main model/optimizer state (Accelerate-managed) ---
#     try:
#         accelerator.load_state(checkpoint_path)
#     except RuntimeError as e:
#         # Try to recover: attempt interpolation again using the model on device and retry load.
#         accelerator.print(f"Accelerator.load_state failed: {e}. Attempting interpolation+retry.")
#         if model is not None and state_dict_path.exists():
#             try:
#                 sd = torch.load(state_dict_path, map_location=accelerator.device)
#                 sd, updated = interpolate_positional_embeddings(sd, model.state_dict())
#                 if updated:
#                     torch.save(sd, state_dict_path)
#                     accelerator.print("✅ Retried: saved resized positional embeddings into checkpoint.")
#             except Exception as e2:
#                 accelerator.print(f"Retry preprocessing failed: {e2}")
#         # last attempt (let exception bubble if it still fails)
#         accelerator.load_state(checkpoint_path)

#     # --- Handle positional embeddings if model is provided ---
#     if model is not None:
#         # Load raw state_dict if available
#         state_dict_path = checkpoint_path / "pytorch_model.bin"
#         if state_dict_path.exists():
#             state_dict = torch.load(state_dict_path, map_location=accelerator.device)
#             model_state = model.state_dict()

#             state_dict, updated = interpolate_positional_embeddings(state_dict, model_state)
#             model.load_state_dict(state_dict, strict=False)
#             if updated:
#                 accelerator.print("✅ Positional embeddings resized to match new resolution.")
#         else:
#             accelerator.print("⚠️ No explicit model state_dict found; skipping interpolation.")

#     # --- Load EMA if available ---
#     ema_path = checkpoint_path / "ema.pt"
#     if ema is not None and ema_path.exists():
#         ema_state = torch.load(ema_path, map_location=accelerator.device)
        
#         # Interpolate positional embeddings in EMA state if model is provided
#         if model is not None and 'shadow_params' in ema_state:
#             model_state = model.state_dict()
#             ema_shadow = ema_state['shadow_params']
            
#             # EMA shadow_params is a list of tensors, convert to dict for interpolation
#             if isinstance(ema_shadow, list):
#                 # Get parameter names from model
#                 param_names = list(model_state.keys())
#                 if len(ema_shadow) == len(param_names):
#                     # Create a dict from shadow params
#                     ema_shadow_dict = {name: param for name, param in zip(param_names, ema_shadow)}
                    
#                     # Interpolate
#                     ema_shadow_dict, updated = interpolate_positional_embeddings(ema_shadow_dict, model_state)
                    
#                     # Convert back to list
#                     ema_state['shadow_params'] = [ema_shadow_dict[name] for name in param_names]
                    
#                     if updated:
#                         accelerator.print("✅ EMA positional embeddings resized to match new resolution.")
#                 else:
#                     accelerator.print(f"⚠️ EMA shadow params length mismatch: {len(ema_shadow)} vs {len(param_names)}")
#             else:
#                 # If it's already a dict, interpolate directly
#                 ema_shadow, updated = interpolate_positional_embeddings(ema_shadow, model_state)
#                 ema_state['shadow_params'] = ema_shadow
                
#                 if updated:
#                     accelerator.print("✅ EMA positional embeddings resized to match new resolution.")
        
#         ema.load_state_dict(ema_state)
#         accelerator.print("✅ EMA weights loaded.")

#     # --- Re-initialize EMA if needed to match model parameter shapes ---
#     if ema is not None and model is not None:
#         from torch_ema import ExponentialMovingAverage
#         # Get decay from existing EMA or default
#         decay = getattr(ema, 'decay', 0.999)
#         device = next(model.parameters()).device
#         # Re-create EMA object with current model parameters
#         ema.__init__(model.parameters(), decay=decay)
#         accelerator.print("✅ EMA object re-initialized to match model parameter shapes.")

#     accelerator.print("Checkpoint loaded successfully.")