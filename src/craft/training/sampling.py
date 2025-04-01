"""
Standalone functions for text generation using sampling methods.
"""

import torch
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Any, Optional

def generate_text_sampling(model, char_to_idx, idx_to_char, seed_text, max_length=500, temperature=0.8, 
                 device=None, top_k=0, top_p=0.0, repetition_penalty=1.0):
    """
    Generate text from a trained model using sampling.
    
    Args:
        model: Trained model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        seed_text: Initial text to condition the generation
        max_length: Maximum number of characters to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to run generation on
        top_k: Limit sampling to top k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (0.0 = disabled)
        repetition_penalty: Penalty for repeating tokens
        
    Returns:
        Generated text including the seed
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    generated = seed_text
    
    # Convert seed text to indices
    context = [char_to_idx.get(c, char_to_idx.get("<unk>", 0)) for c in seed_text]
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            outputs = model(context)
            
            # Focus on the last token predictions
            # Assume outputs are logits or tuple where first element is logits
            if isinstance(outputs, tuple):
                next_token_logits = outputs[0][:, -1, :] 
            else: # Assuming outputs are just logits
                next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for prev_token in context[0]:
                    # Only apply penalty if token_id is valid
                    if 0 <= prev_token.item() < next_token_logits.size(-1):
                         next_token_logits[0, prev_token.item()] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                # Ensure top_k is not larger than vocab size
                effective_top_k = min(top_k, next_token_logits.size(-1))
                top_k_vals, _ = torch.topk(next_token_logits, effective_top_k)
                indices_to_remove = next_token_logits < top_k_vals[:, [-1]] 
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                if sorted_indices_to_remove.shape[-1] > 1:
                     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to generated text
            # Handle potential KeyError if next_token is out of bounds for idx_to_char
            next_char = idx_to_char.get(next_token, '<UNK>') # Use <UNK> or similar for safety
            generated += next_char
            
            # Update context for next prediction
            context = torch.cat((context, torch.tensor([[next_token]], device=device)), dim=1)
            
            # Optionally truncate context to save memory for long generations
            model_max_length = getattr(model.config, 'max_seq_length', 1024) if hasattr(model, 'config') else 1024 # Use max_seq_length if available
            if context.size(1) > model_max_length:
                context = context[:, -model_max_length:]
            
            # Check for EOS token using the *character* mapping if applicable
            # eos_char = idx_to_char.get(char_to_idx.get("<eos>", -1), None)
            # if eos_char and next_char == eos_char:
            #     break
    
    return generated

def sample_text(model, char_to_idx, idx_to_char, num_samples=5, seed_text="TYRION: ", 
               max_length=500, temperature=0.8, device=None, log_samples=True, **kwargs):
    """
    Generate multiple text samples from a trained model using sampling.
    
    Args:
        model: Trained model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        num_samples: Number of samples to generate
        seed_text: Initial text to condition the generation
        max_length: Maximum number of characters per sample
        temperature: Sampling temperature (higher = more random)
        device: Device to run generation on
        log_samples: Whether to log samples
        **kwargs: Additional keyword arguments passed to generate_text 
                  (e.g., top_k, top_p, repetition_penalty).
        
    Returns:
        List of generated samples
    """
    samples = []
    
    if log_samples:
        logging.info(f"Generating {num_samples} samples with temperature {temperature}...")
        logging.info(f"Seed text: '{seed_text}'")
    
    for i in range(num_samples):
        start_time = time.time()
        
        # Generate the sample, passing through extra kwargs
        sample = generate_text_sampling(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
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
            # Avoid division by zero if generation was instant
            chars_per_sec = len(sample) / generation_time if generation_time > 0 else float('inf')
            
            logging.info(f"\nSample {i+1}/{num_samples} (generated in {generation_time:.2f}s, {chars_per_sec:.1f} char/s):")
            logging.info(f"{'-' * 40}")
            logging.info(sample)
            logging.info(f"{'-' * 40}\n")
    
    return samples 