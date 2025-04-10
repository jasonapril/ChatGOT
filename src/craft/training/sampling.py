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
    top_p: float = 1.0,
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

    # Ensure initial_ids is always a list before extending
    initial_ids = context.squeeze().tolist()
    if not isinstance(initial_ids, list):
        initial_ids = [initial_ids] # Wrap scalar in a list
    generated_token_ids.extend(initial_ids) # Store initial context IDs

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
                        # Apply penalty: divide positive logits, multiply negative logits
                        if next_token_logits[0, prev_token_id.item()] > 0:
                            next_token_logits[0, prev_token_id.item()] /= repetition_penalty
                        else:
                            next_token_logits[0, prev_token_id.item()] *= repetition_penalty # Multiply negative logits

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
    prompts: Optional[List[str]] = None,
    seed_text: str = "The ",
    max_new_tokens: int = 500,
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
        num_samples: Number of samples to generate. If prompts is provided and num_samples is not explicitly set, defaults to len(prompts).
        prompts: List of prompts to condition the generation.
        seed_text: Initial text to condition the generation if prompts is None.
        max_new_tokens: Maximum number of *new* characters/tokens per sample.
        temperature: Sampling temperature (higher = more random).
        device: Device to run generation on.
        log_samples: Whether to log samples.
        **kwargs: Additional keyword arguments passed to generate_text_manual_sampling
                  (e.g., top_k, top_p, repetition_penalty).

    Returns:
        List of generated samples (including the seed text).
    """
    samples = []

    if not hasattr(tokenizer, 'encode'):
        logger.error("Tokenizer must have an 'encode' method to calculate max_length.")
        return [] # Or raise error

    # Determine the actual number of samples to generate
    if prompts and num_samples == 5: # Check against the actual default value
        effective_num_samples = len(prompts)
    else:
        effective_num_samples = num_samples

    # Determine the prompts to use
    actual_prompts = prompts if prompts else [seed_text] * effective_num_samples

    # Ensure we generate the requested number of samples, even if prompts list is shorter/longer
    if prompts:
        if len(actual_prompts) < effective_num_samples:
            actual_prompts.extend([actual_prompts[-1]] * (effective_num_samples - len(actual_prompts))) # Pad if needed
        elif len(actual_prompts) > effective_num_samples:
            actual_prompts = actual_prompts[:effective_num_samples] # Truncate if needed

    if log_samples:
        logger.info(f"Generating {effective_num_samples} samples via manual sampling with temperature {temperature}...")
        # Log first prompt only if using default seed_text for all
        log_prompt = seed_text if prompts is None else "(Using provided prompts)"
        logger.info(f"Seed text: '{log_prompt}'")

    # Iterate through the actual prompts to use
    for i, current_prompt in enumerate(actual_prompts):
        start_time = time.time()

        # Calculate total max_length for the inner function
        try:
            # Note: assumes encode returns a list or object with __len__
            prompt_tokens = tokenizer.encode(current_prompt)
            prompt_len = len(prompt_tokens)
        except Exception as e:
            logger.warning(f"Could not encode prompt to determine length, using max_new_tokens as total length. Error: {e}")
            prompt_len = 0 # Fallback or default?

        total_max_length = prompt_len + max_new_tokens

        # Generate the sample, passing through extra kwargs, use current_prompt
        sample = generate_text_manual_sampling(
            model=model,
            tokenizer=tokenizer,
            seed_text=current_prompt, # Use the prompt from the list
            max_length=total_max_length, # Pass calculated total length
            temperature=temperature,
            device=device,
            **kwargs # Pass top_k, top_p, etc.
        )

        samples.append(sample)

        # Log the sample if requested
        if log_samples:
            generation_time = time.time() - start_time
            # Calculate length excluding seed text for meaningful throughput
            generated_part_len = (len(sample) - len(current_prompt)) if isinstance(sample, str) else 0
            chars_per_sec = generated_part_len / generation_time if generation_time > 0 else float('inf')

            logger.info(f"\nSample {i+1}/{effective_num_samples} (generated {generated_part_len} new tokens in {generation_time:.2f}s, {chars_per_sec:.1f} tokens/s):")
            logger.info(f"{'-' * 40}")
            logger.info(sample)
            logger.info(f"{'-' * 40}\n")

    return samples 