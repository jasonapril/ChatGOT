#!/usr/bin/env python
"""
Generate text samples from trained ChatGoT models

This script loads a trained GPTDecoder model checkpoint
and generates text samples based on a given prompt.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add the project root to the path so we can import the src package
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import CharDataset
from src.models.gpt_decoder import create_gpt_model
from src.utils.generation import generate_sample_text

def setup_logging():
    """Configure logging to console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained ChatGoT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--prompt", type=str, default="TYRION: ", help="Prompt text for generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=float, default=40, help="Top-k sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=250, help="Maximum number of tokens to generate")
    parser.add_argument("--data_path", type=str, default="data/raw/got/game_of_thrones.txt",
                       help="Path to the training data (needed for character mapping)")
    parser.add_argument("--output_dir", type=str, default="outputs/samples",
                       help="Directory to save generated samples")
    args = parser.parse_args()
    
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset for tokenization
    logger.info(f"Loading dataset from {args.data_path}")
    try:
        block_size = 128  # Default, might be overridden by checkpoint config
        
        # Load the dataset text directly
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create dataset instance
        dataset = CharDataset(text, block_size)
        
        # Get vocabulary size
        vocab_size = len(dataset.char_to_idx)
        logger.info(f"Dataset loaded with vocabulary size: {vocab_size}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.exception(e)
        return
    
    # Load the model checkpoint
    logger.info(f"Loading model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        config = checkpoint.get("config", {})
        
        # Extract model parameters from config
        model_vocab_size = config.get("vocab_size", vocab_size)
        d_model = config.get("d_model", 512)
        n_head = config.get("n_head", 8)
        n_layers = config.get("n_layers", 8)
        max_seq_length = config.get("n_positions", 256)
        d_hid = config.get("d_hid", 2048)  # Use 2048 to match trained model
        
        logger.info(f"Model config: vocab_size={model_vocab_size}, d_model={d_model}, "
                   f"n_head={n_head}, n_layers={n_layers}, d_hid={d_hid}, max_seq_length={max_seq_length}")
        
        # Create the model with the same architecture as in training
        model = create_gpt_model(
            vocab_size=model_vocab_size,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            d_hid=d_hid,
            max_seq_length=max_seq_length
        )
        
        # Load state dict with key mapping
        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        new_state_dict = {}
        
        # Map key names from checkpoint format to model format
        key_mapping = {
            'norm.weight': 'ln_f.weight',
            'norm.bias': 'ln_f.bias',
            'output_layer.weight': 'out_proj.weight',
            'output_layer.bias': 'out_proj.bias'
        }
        
        for key, value in model_state_dict.items():
            # Handle key name differences
            new_key = key
            for old_pattern, new_pattern in key_mapping.items():
                if old_pattern in key:
                    new_key = key.replace(old_pattern, new_pattern)
                    break
                    
            # Skip causal mask buffers which are generated by the model
            if 'causal_mask' in key:
                continue
                
            new_state_dict[new_key] = value
        
        # Load the mapped state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.exception(e)
        return
    
    # Generate text with the given prompt
    logger.info(f"Generating text with prompt: '{args.prompt}'")
    try:
        # Convert the prompt to token ids
        context = torch.tensor([dataset.char_to_idx.get(c, 0) for c in args.prompt], 
                              dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate sample text
        sample = generate_sample_text(
            model=model,
            context=context,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            tokenizer=dataset,
            verbose=True
        )
        
        # Get the newly generated part (excluding the prompt)
        generated_text = sample[len(args.prompt):]
        
        # Save the generated text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/sample_{timestamp}.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {args.prompt}\n\n")
            f.write(f"Generated text:\n{generated_text}\n")
        
        logger.info(f"Generated text saved to {output_path}")
        
        # Print the generated text
        logger.info("\nGenerated sample:")
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Generated: {generated_text}")
        
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        logger.exception(e)

if __name__ == "__main__":
    main() 