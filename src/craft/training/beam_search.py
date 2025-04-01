"""
Standalone function for text generation using Beam Search.
"""

import torch
import torch.nn.functional as F
import logging
import time
from typing import Dict, List, Any, Optional

def beam_search_generate(model, char_to_idx, idx_to_char, seed_text, max_length=500, 
                        beam_width=5, device=None, length_penalty=1.0):
    """
    Generate text using beam search for higher quality.
    
    Args:
        model: Trained model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        seed_text: Initial text to condition the generation
        max_length: Maximum number of characters to generate
        beam_width: Number of beams to track
        device: Device to run generation on
        length_penalty: Penalty applied to longer sequences
        
    Returns:
        Generated text with highest score
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Convert seed text to indices
    context = [char_to_idx.get(c, char_to_idx.get("<unk>", 0)) for c in seed_text]
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    
    # Initialize beams with seed
    beams = [(context, 0.0, seed_text)]  # (tensor, score, text)
    
    with torch.no_grad():
        for _ in range(max_length):
            candidates = []
            
            # Process each beam
            for beam_context, beam_score, beam_text in beams:
                # Get predictions
                outputs = model(beam_context)
                # Assume outputs are logits or tuple where first element is logits
                if isinstance(outputs, tuple):
                    next_token_logits = outputs[0][:, -1, :] 
                else: # Assuming outputs are just logits
                    next_token_logits = outputs[:, -1, :]
                
                log_probs = F.log_softmax(next_token_logits.squeeze(0), dim=-1)
                
                # Get top-k candidates for each beam
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for log_prob, token_idx in zip(topk_log_probs, topk_indices):
                    token_idx = token_idx.item()
                    new_context = torch.cat((beam_context, torch.tensor([[token_idx]], device=device)), dim=1)
                    
                    # Compute new score (average log prob)
                    token_score = log_prob.item()
                    # Apply length penalty: Score = sum(log_probs) / len**penalty
                    # Need to recalculate sum, current score is already normalized
                    # Let's just add log_prob for now - simpler, standard beam search scoring
                    new_score = beam_score + token_score 
                    
                    # Handle potential KeyError
                    new_text = beam_text + idx_to_char.get(token_idx, '<UNK>')
                    candidates.append((new_context, new_score, new_text))
            
            # Sort and keep top beams based on score (sum log_probs)
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Optional: Check for end-of-sequence token in top beam
            top_context, _, top_text = beams[0]
            last_token_id = top_context[0, -1].item()
            if last_token_id == char_to_idx.get("<eos>", -1): # Use a non-existent index if EOS not in vocab
                break # Stop generation if top beam ends

            # Optionally truncate contexts to save memory
            model_max_length = getattr(model.config, 'max_seq_length', 1024) if hasattr(model, 'config') else 1024
            beams = [(context[:, -model_max_length:] if context.size(1) > model_max_length else context, 
                     score, text) for context, score, text in beams]
    
    # Apply length penalty *after* generation is complete for final ranking
    final_beams = []
    for context, score, text in beams:
        final_score = score / (len(text) ** length_penalty)
        final_beams.append((context, final_score, text))
        
    # Return the top beam after length penalty normalization
    final_beams = sorted(final_beams, key=lambda x: x[1], reverse=True)
    return final_beams[0][2] 