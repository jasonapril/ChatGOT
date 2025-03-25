"""Text generation utilities for ChatGoT."""
import torch
import torch.nn.functional as F
from typing import Optional

def generate_text(
    model: torch.nn.Module,
    tokenizer: any,  # Any tokenizer with encode/decode methods
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: str = "cuda",
    seed_text: Optional[str] = None,
) -> str:
    """
    Generate text using the trained model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer for encoding/decoding text
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider for sampling
        top_p: Cumulative probability threshold for sampling
        device: Device to run generation on
        seed_text: Optional seed text to start generation from
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Set up seed text
    if seed_text is None:
        seed_text = "The"
    
    # Encode seed text
    input_ids = tokenizer.encode(seed_text)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # Decode generated text
    generated_text = tokenizer.decode(input_ids[0].tolist())
    
    return generated_text 