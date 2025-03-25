#!/usr/bin/env python
"""
Compare parameter counts between standard transformer and optimized GPT decoder.
"""

import sys
import torch
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the model implementations
from src.models.transformer import TransformerModel
from src.models.gpt_decoder import GPTDecoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def main():
    # Model configuration
    config = {
        'vocab_size': 96,
        'd_model': 768,
        'n_head': 12,
        'd_hid': 3072,
        'n_layers': 12,
        'dropout': 0.1,
        'max_seq_length': 1024,
        'layer_norm_eps': 1e-5,
        'activation': 'gelu',
        'bias': True
    }
    
    # Create models
    logger.info("Creating standard transformer model with cross-attention...")
    transformer_model = TransformerModel(**config)
    
    logger.info("Creating optimized GPT decoder without cross-attention...")
    gpt_decoder_model = GPTDecoder(**config)
    
    # Count parameters
    transformer_params = count_parameters(transformer_model)
    gpt_decoder_params = count_parameters(gpt_decoder_model)
    
    # Calculate difference
    param_diff = transformer_params - gpt_decoder_params
    param_reduction_percent = (param_diff / transformer_params) * 100
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("MODEL PARAMETER COMPARISON")
    logger.info("="*60)
    logger.info(f"Standard Transformer:  {transformer_params:,} parameters")
    logger.info(f"Optimized GPT Decoder: {gpt_decoder_params:,} parameters")
    logger.info(f"Difference:            {param_diff:,} parameters ({param_reduction_percent:.1f}% reduction)")
    
    # Print model component parameter counts for GPT decoder
    logger.info("\nGPT Decoder Parameter Breakdown:")
    logger.info(f"  Token Embedding:      {count_parameters(gpt_decoder_model.token_embedding):,}")
    logger.info(f"  Position Embedding:   {count_parameters(gpt_decoder_model.position_embedding):,}")
    
    total_layer_params = 0
    for i, layer in enumerate(gpt_decoder_model.layers):
        layer_params = count_parameters(layer)
        total_layer_params += layer_params
        
    logger.info(f"  Decoder Layers Total: {total_layer_params:,}")
    logger.info(f"  Final Layer Norm:     {count_parameters(gpt_decoder_model.norm):,}")
    logger.info(f"  Output Layer:         {count_parameters(gpt_decoder_model.output_layer):,}")
    
    # Print model component parameter counts for Transformer
    logger.info("\nTransformer Parameter Breakdown:")
    logger.info(f"  Token Embedding:      {count_parameters(transformer_model.token_embedding):,}")
    logger.info(f"  Position Embedding:   {count_parameters(transformer_model.position_embedding):,}")
    
    total_decoder_params = count_parameters(transformer_model.transformer_decoder)
    logger.info(f"  Transformer Decoder:  {total_decoder_params:,}")
    logger.info(f"  Output Layer:         {count_parameters(transformer_model.output_layer):,}")

if __name__ == "__main__":
    main() 