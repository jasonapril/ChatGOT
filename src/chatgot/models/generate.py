"""Text generation module for ChatGoT."""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from chatgot.core.config import get_full_path
from chatgot.models.model_io import load_model
from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


def encode_text(text: str, vocab_size: int = 256) -> torch.Tensor:
    """
    Encode text as a tensor of character indices for character-level models.
    
    Args:
        text: Text to encode
        vocab_size: Vocabulary size (assumed to be character-level, 0-255)
        
    Returns:
        Tensor of character indices
    """
    # For character-level models, we can just use the ASCII/Unicode values
    encoded = torch.tensor([ord(c) % vocab_size for c in text], dtype=torch.long)
    # Add batch dimension
    return encoded.unsqueeze(0)


def decode_text(tokens: torch.Tensor, strip_padding: bool = True) -> str:
    """
    Decode a tensor of character indices back to text.
    
    Args:
        tokens: Tensor of token indices [batch_size, seq_len]
        strip_padding: Whether to strip padding tokens (0)
        
    Returns:
        Decoded text
    """
    # Take the first batch
    if tokens.dim() > 1:
        tokens = tokens[0]
    
    # Convert to list and then to characters
    token_list = tokens.tolist()
    if strip_padding:
        token_list = [t for t in token_list if t > 0]
    
    return ''.join([chr(t) for t in token_list])


def generate_text(
    cfg: DictConfig,
    checkpoint_path: Optional[Union[str, Path]] = None,
    prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    output_file: Optional[Union[str, Path]] = None,
    num_samples: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Generate text using a trained model.
    
    Args:
        cfg: Configuration
        checkpoint_path: Path to model checkpoint
        prompt: Text prompt to start generation with
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for repeating tokens
        output_file: File to write generated text to
        num_samples: Number of text samples to generate
        device: Device to run inference on
        seed: Random seed for reproducibility
        
    Returns:
        List of generated text samples
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")
    
    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get parameters from config if not explicitly provided
    if checkpoint_path is None:
        checkpoint_path = cfg.generate.get("checkpoint_path", "${paths.models_dir}/latest.pt")
    
    checkpoint_path = get_full_path(checkpoint_path, cfg)
    
    if prompt is None:
        prompt = cfg.generate.get("prompt", "")
    
    if max_new_tokens is None:
        max_new_tokens = cfg.generate.get("max_new_tokens", 100)
    
    if temperature is None:
        temperature = cfg.generate.get("temperature", 1.0)
    
    if top_k is None:
        top_k = cfg.generate.get("top_k", 0)
    
    if top_p is None:
        top_p = cfg.generate.get("top_p", 0.9)
    
    if repetition_penalty is None:
        repetition_penalty = cfg.generate.get("repetition_penalty", 1.2)
    
    if output_file is None:
        output_file = cfg.generate.get("output_file", "${paths.outputs_dir}/generated_text.txt")
    
    output_file = get_full_path(output_file, cfg)
    
    if num_samples is None:
        num_samples = cfg.generate.get("num_samples", 1)

    # Load model
    checkpoint = load_model(
        load_path=checkpoint_path,
        config=cfg,
        device=device
    )
    model = checkpoint["model"]
    model.eval()
    
    # Log generation parameters
    logger.info(f"Generating text with parameters:")
    logger.info(f"  - Prompt: {prompt[:30]}{'...' if len(prompt) > 30 else ''}")
    logger.info(f"  - Max new tokens: {max_new_tokens}")
    logger.info(f"  - Temperature: {temperature}")
    logger.info(f"  - Top-k: {top_k}")
    logger.info(f"  - Top-p: {top_p}")
    logger.info(f"  - Repetition penalty: {repetition_penalty}")
    logger.info(f"  - Number of samples: {num_samples}")
    
    # Generate samples
    generated_texts = []
    
    for i in range(num_samples):
        logger.info(f"Generating sample {i+1}/{num_samples}...")
        
        # Encode prompt
        input_ids = encode_text(prompt, model.vocab_size).to(device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                verbose=(i == 0)  # Only be verbose for the first sample
            )
        
        # Decode generated tokens
        generated_text = decode_text(generated_ids)
        generated_texts.append(generated_text)
        
        # Log a preview
        preview_length = min(100, len(generated_text))
        logger.info(f"Generated text (preview): {generated_text[:preview_length]}...")
    
    # Save to file if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, text in enumerate(generated_texts):
                f.write(f"=== Sample {i+1} ===\n\n")
                f.write(text)
                f.write("\n\n")
        
        logger.info(f"Generated texts saved to {output_file}")
    
    return generated_texts 