import torch
import numpy as np


def compute_token_usage(tokens):
    """
    Compute the token usage as the ratio of unique tokens to total tokens.
    
    Args:
        tokens: Can be a single tensor or a list/tuple of token tensors from hierarchical quantization.
                Each tensor should have shape [B, ...] where tokens are discrete integer indices.
    
    Returns:
        usage (float): Percentage of unique tokens used (0-100)
        unique_count (int): Number of unique tokens
        total_count (int): Total number of tokens
    """
    if tokens is None:
        return 0.0, 0, 0
    
    # Handle hierarchical tokens (list/tuple of tensors)
    if isinstance(tokens, (list, tuple)):
        all_tokens = []
        for token_tensor in tokens:
            if token_tensor is not None:
                if torch.is_tensor(token_tensor):
                    all_tokens.append(token_tensor.detach().cpu().flatten())
                else:
                    # Handle nested lists/tuples
                    for t in token_tensor:
                        if t is not None and torch.is_tensor(t):
                            all_tokens.append(t.detach().cpu().flatten())
        
        if len(all_tokens) == 0:
            return 0.0, 0, 0
        
        # Concatenate all tokens
        tokens_flat = torch.cat(all_tokens)
    else:
        # Single tensor
        if not torch.is_tensor(tokens):
            return 0.0, 0, 0
        tokens_flat = tokens.detach().cpu().flatten()
    
    # Compute unique tokens
    unique_tokens = torch.unique(tokens_flat)
    unique_count = len(unique_tokens)
    total_count = len(tokens_flat)
    
    if total_count == 0:
        return 0.0, 0, 0
    
    usage_percentage = (unique_count / total_count) * 100.0
    
    return usage_percentage, unique_count, total_count


def compute_codebook_usage(tokens, codebook_size):
    """
    Compute what fraction of the codebook is actually used.
    
    Args:
        tokens: Token tensor(s) as in compute_token_usage
        codebook_size (int): Total size of the codebook
    
    Returns:
        codebook_usage (float): Percentage of codebook entries used (0-100)
        unique_count (int): Number of unique tokens observed
    """
    _, unique_count, _ = compute_token_usage(tokens)
    
    if codebook_size == 0:
        return 0.0, unique_count
    
    codebook_usage_percentage = (unique_count / codebook_size) * 100.0
    
    return codebook_usage_percentage, unique_count
