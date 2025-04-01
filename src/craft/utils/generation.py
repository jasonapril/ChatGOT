"""
Text generation utilities for language models.
"""
import torch
import logging
from typing import Optional, Union, List, Tuple, Any

def generate_sample_text(
    model: torch.nn.Module,
    context: torch.Tensor,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    tokenizer: Any = None,
    verbose: bool = False
) -> str:
    """
    Generate sample text from the model.
    
    Args:
        model: The model to generate text with
        context: The context to use for generation, shape [batch_size, seq_len]
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Number of highest probability tokens to keep for top-k sampling
        top_p: Probability threshold for top-p sampling
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        tokenizer: Tokenizer or dataset with decode method
        verbose: Whether to log progress
        
    Returns:
        Generated text
    """
    # Generate token ids
    model.eval()
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            verbose=verbose
        )
    
    # Decode the generated tokens
    if hasattr(tokenizer, 'decode'):
        sample_text = tokenizer.decode(output_ids[0].tolist())
    else:
        sample_text = tokenizer.decode(output_ids[0])
    
    return sample_text


def sample_text(
    model: torch.nn.Module,
    prompt: str = "",
    max_length: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    tokenizer: Any = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Sample text from the model using a prompt.
    
    Args:
        model: The model to generate text with
        prompt: The prompt text
        max_length: Maximum length of the generated text
        temperature: Sampling temperature
        top_k: Number of highest probability tokens to keep for top-k sampling
        top_p: Probability threshold for top-p sampling
        tokenizer: Tokenizer for encoding/decoding
        device: Device to put tensors on
        
    Returns:
        Generated text
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    # Encode the prompt
    if prompt:
        if hasattr(tokenizer, 'char_to_idx'):
            # Character-level dataset
            encoded = torch.tensor(
                [tokenizer.char_to_idx.get(c, 0) for c in prompt],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)
        else:
            # Regular tokenizer
            encoded = tokenizer.encode(prompt, return_tensors='pt').to(device)
    else:
        # Start with an empty tensor
        encoded = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encoded,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode the generated tokens
    if hasattr(tokenizer, 'decode'):
        output_text = tokenizer.decode(output_ids[0].tolist())
    else:
        output_text = tokenizer.decode(output_ids[0])
    
    return output_text


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