#!/usr/bin/env python
"""
Simple test script to generate text from the trained model
"""
import os
import sys
import logging
import torch
from torch import nn
import argparse
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the model from test_train.py
try:
    from test_train import SimpleModel
except ImportError:
    # Define the model here as fallback
    class SimpleModel(nn.Module):
        """A simple model for character-level prediction."""
        
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            self.fc_out = nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, x):
            # Embed the input
            x = self.embedding(x)
            
            # Apply LSTM
            lstm_out, _ = self.lstm(x)
            
            # Project to vocabulary size
            logits = self.fc_out(lstm_out)
            return logits

def generate_text(model, char2idx, idx2char, start_text="The", max_length=100, temperature=1.0, device="cuda"):
    """Generate text using the model."""
    model.eval()
    
    # Convert start text to indices
    indices = [char2idx[char] for char in start_text]
    
    # Create input tensor
    input_seq = torch.tensor([indices], dtype=torch.long).to(device)
    
    generated_text = start_text
    
    # Generate one character at a time
    for _ in range(max_length):
        with torch.no_grad():
            # Forward pass
            output = model(input_seq)
            
            # Get the last time step output
            logits = output[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Convert to character
            next_char = idx2char[str(next_char_idx)]
            
            # Add to generated text
            generated_text += next_char
            
            # Update input sequence for next iteration
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)
    
    return generated_text

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text from trained model")
    parser.add_argument("--model", type=str, default="models/test_model.pt", help="Path to model file")
    parser.add_argument("--start-text", type=str, default="The", help="Starting text for generation")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Determine device
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the trained model
    try:
        logger.info(f"Loading model from {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        
        # Extract vocabulary and configuration
        vocab = checkpoint['vocab']
        config = checkpoint['config']
        
        # Create model
        model = SimpleModel(
            vocab_size=vocab.get('vocab_size', len(vocab['char_to_idx'])),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Get vocabulary mappings
        char2idx = vocab['char_to_idx']
        idx2char = vocab['idx_to_char']
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Generate text
    logger.info(f"Generating text with start: '{args.start_text}', length: {args.max_length}, temperature: {args.temperature}")
    generated_text = generate_text(
        model=model,
        char2idx=char2idx,
        idx2char=idx2char,
        start_text=args.start_text,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device
    )
    
    logger.info("Generated text:")
    print(generated_text)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 