"""
Standalone function for generating text for multiple prompts in parallel.
"""

import torch
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Any, Optional

def batch_generate(model, char_to_idx, idx_to_char, prompts, max_length=500, temperature=0.8, 
                  device=None, top_p=0.9):
    """
    Generate text for multiple prompts in parallel for efficiency.
    
    Args:
        model: Trained model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        prompts: List of seed texts
        max_length: Maximum number of characters to generate
        temperature: Sampling temperature
        device: Device to run generation on
        top_p: Nucleus sampling threshold
        
    Returns:
        List of generated texts
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    batch_size = len(prompts)
    
    # Convert prompts to token IDs
    batch_context = []
    max_prompt_len = 0
    
    for prompt in prompts:
        context = [char_to_idx.get(c, char_to_idx.get("<unk>", 0)) for c in prompt]
        batch_context.append(context)
        max_prompt_len = max(max_prompt_len, len(context))
    
    # Pad contexts to the same length
    pad_id = char_to_idx.get("<pad>", 0) # Use <pad> if available, else 0
    padded_contexts = []
    for context in batch_context:
        padded = context + [pad_id] * (max_prompt_len - len(context))
        padded_contexts.append(padded)
    
    # Convert to tensor
    contexts = torch.tensor(padded_contexts, dtype=torch.long, device=device)
    
    # Track which samples are still generating
    active = [True] * batch_size
    generated = list(prompts)
    eos_token_id = char_to_idx.get("<eos>", -1) # Use a non-existent ID if EOS not in vocab
    
    with torch.no_grad():
        for _ in range(max_length):
            if not any(active):
                break
                
            # Get predictions for active samples only
            # Note: This implementation recomputes for the whole active batch each time.
            # More efficient implementations use KV caching.
            active_indices = [i for i, is_active in enumerate(active) if is_active]
            active_contexts = contexts[active_indices]
            
            outputs = model(active_contexts)
            
            # Assume outputs are logits or tuple where first element is logits
            if isinstance(outputs, tuple):
                all_next_token_logits = outputs[0][:, -1, :] 
            else: # Assuming outputs are just logits
                all_next_token_logits = outputs[:, -1, :]

            # Apply temperature
            all_next_token_logits = all_next_token_logits / temperature
            
            # Apply top-p sampling (batched)
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(all_next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                if sorted_indices_to_remove.shape[-1] > 1:
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter the indices to remove back to original shape
                indices_to_remove = torch.zeros_like(all_next_token_logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                all_next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token for each active sequence
            probs = F.softmax(all_next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1) # [num_active, 1]
            
            # Process results for each active sequence
            current_active_idx = 0
            for i in range(batch_size):
                if active[i]:
                    next_token = next_tokens[current_active_idx].item()
                    
                    # Check for end of generation 
                    if next_token == eos_token_id:
                        active[i] = False
                    else:
                        # Add to generated text
                        generated[i] += idx_to_char.get(next_token, '<UNK>')
                        
                        # Update context (append and potentially truncate)
                        # Note: Simple append without KV cache - becomes inefficient for long sequences
                        new_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                        contexts[i] = torch.cat((contexts[i:i+1, 1:], new_token_tensor), dim=1) # Shift and append
                        
                    current_active_idx += 1
                    
            # Optionally truncate all contexts to save memory 
            model_max_length = getattr(model.config, 'max_seq_length', 1024) if hasattr(model, 'config') else 1024
            if contexts.size(1) > model_max_length:
                 contexts = contexts[:, -model_max_length:]

    return generated 