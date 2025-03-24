#!/usr/bin/env python
"""
Text Generation Module
====================

This module provides functionality for generating text from trained models:

1. Efficient sampling implementations
2. Various generation strategies
3. Temperature and top-k/p sampling
4. Beam search
5. Batch generation for higher throughput

These functions are optimized for both quality and speed.
"""

import torch
import torch.nn.functional as F
import logging
import time
import random
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

def generate_text(model, char_to_idx, idx_to_char, seed_text, max_length=500, temperature=0.8, 
                 device=None, top_k=0, top_p=0.0, repetition_penalty=1.0):
    """
    Generate text from a trained model.
    
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
            next_token_logits = outputs[0, -1, :].cpu()
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for prev_token in context[0]:
                    next_token_logits[prev_token.item()] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1).item()
            
            # Add to generated text
            generated += idx_to_char[next_token]
            
            # Update context for next prediction
            context = torch.cat((context, torch.tensor([[next_token]], device=device)), dim=1)
            
            # Optionally truncate context to save memory for long generations
            max_context_length = model.max_seq_length if hasattr(model, 'max_seq_length') else 1024
            if context.size(1) > max_context_length:
                context = context[:, -max_context_length:]
    
    return generated

def sample_text(model, char_to_idx, idx_to_char, num_samples=5, seed_text="TYRION: ", 
               max_length=500, temperature=0.8, device=None, log_samples=True):
    """
    Generate multiple text samples from a trained model.
    
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
        
    Returns:
        List of generated samples
    """
    samples = []
    
    if log_samples:
        logging.info(f"Generating {num_samples} samples with temperature {temperature}...")
        logging.info(f"Seed text: '{seed_text}'")
    
    for i in range(num_samples):
        start_time = time.time()
        
        # Generate the sample
        sample = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seed_text=seed_text,
            max_length=max_length,
            temperature=temperature,
            device=device
        )
        
        samples.append(sample)
        
        # Log the sample if requested
        if log_samples:
            generation_time = time.time() - start_time
            chars_per_sec = len(sample) / generation_time
            
            logging.info(f"\nSample {i+1}/{num_samples} (generated in {generation_time:.2f}s, {chars_per_sec:.1f} char/s):")
            logging.info(f"{'-' * 40}")
            logging.info(sample)
            logging.info(f"{'-' * 40}\n")
    
    return samples

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
                next_token_logits = outputs[0, -1, :].cpu()
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k candidates for each beam
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for log_prob, token_idx in zip(topk_log_probs, topk_indices):
                    token_idx = token_idx.item()
                    new_context = torch.cat((beam_context, torch.tensor([[token_idx]], device=device)), dim=1)
                    
                    # Compute new score (average log prob)
                    token_score = log_prob.item()
                    # Apply length penalty
                    sequence_length = new_context.size(1)
                    normalized_score = beam_score + token_score / (sequence_length ** length_penalty)
                    
                    new_text = beam_text + idx_to_char[token_idx]
                    candidates.append((new_context, normalized_score, new_text))
            
            # Sort and keep top beams
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Optionally truncate contexts to save memory
            max_context_length = model.max_seq_length if hasattr(model, 'max_seq_length') else 1024
            beams = [(context[:, -max_context_length:] if context.size(1) > max_context_length else context, 
                     score, text) for context, score, text in beams]
    
    # Return the top beam
    return beams[0][2]

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
    padded_contexts = []
    for context in batch_context:
        padded = context + [char_to_idx.get("<pad>", 0)] * (max_prompt_len - len(context))
        padded_contexts.append(padded)
    
    # Convert to tensor
    contexts = torch.tensor(padded_contexts, dtype=torch.long, device=device)
    
    # Track which samples are still generating
    active = [True] * batch_size
    generated = list(prompts)
    
    with torch.no_grad():
        for _ in range(max_length):
            if not any(active):
                break
                
            # Get predictions for active samples only
            active_indices = [i for i, is_active in enumerate(active) if is_active]
            active_contexts = contexts[active_indices]
            
            outputs = model(active_contexts)
            
            # Process each active sample
            for i, idx in enumerate(active_indices):
                # Get predictions for the last token
                next_token_logits = outputs[i, -1, :].cpu() / temperature
                
                # Apply top-p sampling
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Check for end of generation (e.g., end token)
                if next_token == char_to_idx.get("<eos>", -1):
                    active[idx] = False
                    continue
                
                # Add to generated text
                generated[idx] += idx_to_char[next_token]
                
                # Update context
                contexts[idx] = torch.cat((contexts[idx], torch.tensor([next_token], device=device)))[-max_prompt_len:]
    
    return generated 