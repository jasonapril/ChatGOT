"""
Text generation utilities for language models.
"""
import torch
import logging
from typing import Optional, Union, List, Tuple, Any
from ..models.base import Model

logger = logging.getLogger(__name__)

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """
    Filter logits using top-k and/or top-p (nucleus) filtering.
    
    Args:
        logits: Logits to filter, shape [batch_size, vocab_size]
        top_k: Keep only the top-k tokens with highest probability (top-k filtering)
        top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value: Value to assign to filtered tokens
        
    Returns:
        Filtered logits
    """
    # Clone logits to avoid modifying the original
    logits = logits.clone()
    
    # Apply top-k filtering
    if top_k > 0:
        vocab_size = logits.size(-1)
        # Clamp top_k to vocab size to prevent errors
        actual_top_k = min(top_k, vocab_size)
        if actual_top_k < vocab_size: # Only filter if top_k is less than vocab size
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, actual_top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits 