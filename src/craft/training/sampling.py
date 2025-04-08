"""
Standalone functions for text generation using sampling methods.
"""

import torch
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Any, Optional

# Import utility function
from ..utils.generation import top_k_top_p_filtering
# Import tokenizer base class for type hinting
from ..data.tokenizers.base import Tokenizer

logger = logging.getLogger(__name__)

def generate_text_manual_sampling(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    seed_text: str,
    max_length: int = 500,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0
) -> str:
    """
    Generate text from a trained model using manual sampling logic.
    Uses the provided tokenizer for encoding and decoding.

    Args:
        model: Trained model.
        tokenizer: Initialized tokenizer instance.
        seed_text: Initial text to condition the generation.
        max_length: Maximum number of *new* tokens to generate.
        temperature: Sampling temperature (higher = more random).
        device: Device to run generation on.
        top_k: Limit sampling to top k tokens (0 = disabled).
        top_p: Nucleus sampling threshold (0.0 = disabled).
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).

    Returns:
        Generated text (including the seed text).
    """
    if device is None:
        device = next(model.parameters()).device
    if tokenizer is None or not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'decode'):
        raise ValueError("A valid tokenizer with encode/decode methods must be provided.")

    model.eval()
    generated_token_ids: List[int] = []

    # Encode seed text
    try:
        context_ids = tokenizer.encode(seed_text)
        context = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    except Exception as e:
        logger.error(f"Failed to encode seed text '{seed_text}': {e}", exc_info=True)
        return "[Encoding Error]"

    # Truncate initial context if needed
    model_max_length = getattr(model.config, 'max_seq_length', 1024) if hasattr(model, 'config') else 1024
    if context.size(1) > model_max_length:
        context = context[:, -model_max_length:]
        logger.warning(f"Initial seed text length ({len(context_ids)}) exceeded model max length ({model_max_length}). Truncating.")

    generated_token_ids.extend(context.squeeze().tolist()) # Store initial context IDs

    with torch.no_grad():
        for _ in range(max_length):
            # Ensure context does not exceed model max length before passing to model
            current_context = context[:, -model_max_length:]

            # Get predictions
            outputs = model(current_context)

            # Focus on the last token predictions
            if isinstance(outputs, tuple):
                next_token_logits = outputs[0][:, -1, :] # Logits are often first element
            else:
                next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty > 1.0 and context.numel() > 0: # Check context is not empty
                # Use the full context tensor directly for checking previous tokens
                for prev_token_id in context[0]:
                    if 0 <= prev_token_id.item() < next_token_logits.size(-1):
                         next_token_logits[0, prev_token_id.item()] /= repetition_penalty

            # Apply top-k and top-p filtering using the utility function
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # Convert to probabilities and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token_id = next_token.item()

            # Append to generated sequence and update context
            # Explicitly cast to int to satisfy mypy
            generated_token_ids.append(int(next_token_id))
            context = torch.cat((context, next_token.to(context.device)), dim=1)

            # Check for EOS token
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_token_id is not None and next_token_id == eos_token_id:
                logger.debug(f"EOS token ({eos_token_id}) generated. Stopping generation.")
                break

    # Decode the full sequence of generated IDs
    try:
        # Remove skip_special_tokens=True as base Tokenizer may not support it
        decoded_text = tokenizer.decode(generated_token_ids)
    except Exception as e:
        logger.error(f"Failed to decode generated token IDs: {e}", exc_info=True)
        decoded_text = f"[Decoding Error: {e}]"

    return decoded_text

def generate_samples_manual_sampling(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    num_samples: int = 5,
    seed_text: str = "The ",
    max_length: int = 500,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
    log_samples: bool = True,
    **kwargs: Any
) -> List[str]:
    """
    Generate multiple text samples from a trained model using manual sampling.

    Args:
        model: Trained model.
        tokenizer: Initialized tokenizer instance.
        num_samples: Number of samples to generate.
        seed_text: Initial text to condition the generation.
        max_length: Maximum number of *new* characters/tokens per sample.
        temperature: Sampling temperature (higher = more random).
        device: Device to run generation on.
        log_samples: Whether to log samples.
        **kwargs: Additional keyword arguments passed to generate_text_manual_sampling
                  (e.g., top_k, top_p, repetition_penalty).

    Returns:
        List of generated samples (including the seed text).
    """
    samples = []

    if log_samples:
        logger.info(f"Generating {num_samples} samples via manual sampling with temperature {temperature}...")
        logger.info(f"Seed text: '{seed_text}'")

    for i in range(num_samples):
        start_time = time.time()

        # Generate the sample, passing through extra kwargs
        sample = generate_text_manual_sampling(
            model=model,
            tokenizer=tokenizer,
            seed_text=seed_text,
            max_length=max_length,
            temperature=temperature,
            device=device,
            **kwargs # Pass top_k, top_p, etc.
        )

        samples.append(sample)

        # Log the sample if requested
        if log_samples:
            generation_time = time.time() - start_time
            # Calculate length excluding seed text for meaningful throughput
            generated_part_len = len(sample) - len(seed_text)
            chars_per_sec = generated_part_len / generation_time if generation_time > 0 else float('inf')

            logger.info(f"\nSample {i+1}/{num_samples} (generated {generated_part_len} new tokens in {generation_time:.2f}s, {chars_per_sec:.1f} tokens/s):")
            logger.info(f"{'-' * 40}")
            logger.info(sample)
            logger.info(f"{'-' * 40}\n")

    return samples 