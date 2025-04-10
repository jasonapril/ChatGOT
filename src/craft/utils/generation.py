"""
Text generation utilities for language models.
"""
import torch
import logging
from typing import Optional, Union, List, Tuple, Any
from ..models.base import Model

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """
    Filter logits using top-k and/or top-p (nucleus) filtering.
    Applies top-k first, then top-p to the remaining tokens.

    Args:
        logits: Logits to filter, shape [batch_size, vocab_size]
        top_k: Keep only the top-k tokens with highest probability (top-k filtering)
        top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value: Value to assign to filtered tokens

    Returns:
        Filtered logits
    """
    logits = logits.clone() # Work on a copy
    batch_size, vocab_size = logits.shape

    # Indices to remove, initialized to False
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)

    # --- Top-K Filtering ---
    if top_k is not None and top_k > 0:
        # Cap k at vocab size
        effective_k = min(top_k, vocab_size)
        if effective_k < vocab_size: # Only filter if k makes sense
            # Find the k-th largest logit value
            kth_values = torch.topk(logits, k=effective_k, dim=-1)[0][:, -1, None]
            # Mark logits strictly smaller than the k-th value for removal
            indices_to_remove = logits < kth_values

    # --- Top-P Filtering ---
    # Handle p=0.0 explicitly: keep only the single most probable token
    if top_p is not None and top_p == 0.0:
        # Get the index of the max logit (works even if top-k removed some)
        safe_logits_for_max = torch.where(indices_to_remove, torch.full_like(logits, -float('inf')), logits)
        max_logit_indices = torch.argmax(safe_logits_for_max, dim=-1, keepdim=True)
        # Create a mask to remove everything *except* the max logit index
        p0_indices_to_remove = torch.ones_like(logits, dtype=torch.bool)
        p0_indices_to_remove.scatter_(-1, max_logit_indices, False) # Set the max index to False (don't remove)
        indices_to_remove = p0_indices_to_remove # Override any top-k mask

    elif top_p is not None and 0.0 < top_p < 1.0:
        # Apply top-p filtering on logits *not already removed* by top-k
        safe_logits_for_p = torch.where(indices_to_remove, torch.full_like(logits, -float('inf')), logits)
        sorted_logits, sorted_indices = torch.sort(safe_logits_for_p, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove_p = cumulative_probs > top_p
        # Shift indices to keep the first token above the threshold
        sorted_indices_to_remove_p[..., 1:] = sorted_indices_to_remove_p[..., :-1].clone()
        sorted_indices_to_remove_p[..., 0] = 0 # Always keep the most probable token (among those not removed by top-k)

        # Scatter back to original indices
        indices_to_remove_p = torch.scatter(
            torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, sorted_indices_to_remove_p
        )

        # Combine with top-k removals
        indices_to_remove = indices_to_remove | indices_to_remove_p

    # --- Apply Filter ---
    logits[indices_to_remove] = filter_value
    return logits

# NOTE: This function assumes the model has a `forward` method that returns logits
# and potentially relies on model attributes like `max_seq_length` if not passed explicitly.
# Consider making model interface requirements clearer or passing more args.

def autoregressive_generate(
    model: nn.Module, # Requires forward method
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    # Add max_seq_length explicitly if not reliably available on model?
    # max_seq_length: Optional[int] = None, 
    verbose: bool = False
) -> torch.Tensor:
    """
    Performs standard autoregressive text generation with sampling.

    Args:
        model: The model instance (must have a `forward` method).
               Should ideally have `max_seq_length` attribute if not passed.
        input_ids: Tensor of starting token IDs (batch_size, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Softmax temperature (0 for greedy).
        top_k: Keep only top_k tokens for sampling.
        top_p: Keep smallest set of tokens with cumulative probability >= top_p.
        repetition_penalty: Penalty applied to repeated tokens (1.0 = no penalty).
        eos_token_id: ID of the end-of-sequence token to stop generation.
        verbose: Log progress.

    Returns:
        Tensor containing the input_ids plus the generated tokens.
    """
    device = input_ids.device
    batch_size = input_ids.size(0)

    # Determine max_seq_length (critical for context window)
    # Prefer explicit attribute on model, fallback might be risky
    model_max_seq_length = getattr(model, "max_seq_length", None)
    if model_max_seq_length is None:
         # If not on model, maybe get from config if model has .config?
         model_config = getattr(model, "config", None)
         if model_config:
              model_max_seq_length = getattr(model_config, "max_seq_length", None) 
         if model_max_seq_length is None:
              # Fallback or Error?
              logger.warning("Cannot determine model's max_seq_length. Generation might behave unexpectedly or fail.")
              # Let's assume a large default if unavailable, but this is not ideal
              model_max_seq_length = 2048 # Arbitrary large number
    
    max_seq_length = model_max_seq_length

    # --- Input Truncation --- #
    max_input_len = max_seq_length - max_new_tokens
    if max_input_len < 0:
        logger.warning(
            f"max_new_tokens ({max_new_tokens}) > max_seq_length ({max_seq_length}). "
            f"Cannot generate. Returning original input."
        )
        return input_ids

    if input_ids.size(1) > max_input_len:
        logger.warning(
            f"Input prompt length ({input_ids.size(1)}) > max allowed input length ({max_input_len}). "
            f"Truncating prompt."
        )
        input_ids = input_ids[:, -max_input_len:]
    # ------------------------ #

    generated_tokens = input_ids.clone()
    # Track which sequences in the batch have finished
    stop_generation = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i in range(max_new_tokens):
            # Prepare model input: use the last `max_seq_length` tokens
            current_seq_len = generated_tokens.size(1)
            model_input = generated_tokens[:, -max_seq_length:]

            # Forward pass
            try:
                 logits = model(model_input)
                 # Ensure logits have expected shape [batch, seq_len, vocab_size]
                 if len(logits.shape) != 3 or logits.shape[0] != batch_size or logits.shape[1] != model_input.size(1):
                      raise ValueError(f"Model forward pass returned unexpected logits shape: {logits.shape}")
                 next_token_logits = logits[:, -1, :] # Logits for the next token prediction
            except Exception as e:
                 logger.error(f"Error during model forward pass in generation step {i}: {e}", exc_info=True)
                 # Decide how to handle: return current tokens, raise?
                 raise RuntimeError(f"Model forward pass failed during generation step {i}") from e

            # Apply repetition penalty (only to sequences not yet stopped)
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    if not stop_generation[b]:
                        # Penalize tokens present in the current generated sequence
                        # Use model_input which is correctly windowed for context
                        for token_id in model_input[b]: 
                            # Check bounds before penalizing
                            if 0 <= token_id < next_token_logits.size(-1):
                                if next_token_logits[b, token_id] > 0:
                                    next_token_logits[b, token_id] /= repetition_penalty
                                else:
                                    next_token_logits[b, token_id] *= repetition_penalty

            # --- Sampling Logic --- #            
            if temperature == 0: # Greedy
                 next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                 # Temperature
                 if temperature != 1.0:
                     next_token_logits = next_token_logits / temperature
                 # Top-K
                 if top_k is not None and top_k > 0:
                     indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k, dim=-1)[0][..., -1, None]
                     next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
                 # Top-p
                 if top_p is not None and 0.0 < top_p < 1.0:
                     sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                     sorted_indices_to_remove = cumulative_probs > top_p
                     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                     sorted_indices_to_remove[..., 0] = False
                     indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                     next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
                 # Sample
                 probs = F.softmax(next_token_logits, dim=-1)
                 next_token = torch.multinomial(probs, num_samples=1)
            # ---------------------- #

            # Append generated token, handling stopped sequences
            not_stopped_mask = ~stop_generation.unsqueeze(-1)
            placeholder_token = torch.zeros_like(next_token) # Use 0 as placeholder for stopped seqs
            token_to_append = torch.where(not_stopped_mask, next_token, placeholder_token)
            generated_tokens = torch.cat([generated_tokens, token_to_append], dim=1)

            # Update stop generation flags based on EOS
            if eos_token_id is not None:
                just_stopped = (next_token.squeeze(-1) == eos_token_id) & (~stop_generation)
                stop_generation |= just_stopped

            # Early exit if all sequences have stopped
            if stop_generation.all():
                logger.info(f"All sequences finished generation at step {i+1}/{max_new_tokens}.")
                break
            
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{max_new_tokens} tokens...")
    
    # Return the full generated sequences (including prompt)
    return generated_tokens 