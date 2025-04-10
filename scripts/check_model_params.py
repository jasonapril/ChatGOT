#!/usr/bin/env python
"""
Script to analyze model parameter count in the transformer model.
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add the project root to the path
# project_root = Path(__file__).parent.parent # Commented out to avoid mypy path conflicts
# sys.path.append(str(project_root))

from craft.models.transformer import create_transformer_model
# from craft.data.dataset import load_data # Removed old import
from craft.data.datasets.text_dataset import TextDataset # Added TextDataset import
from craft.data.tokenizers.char import CharTokenizer # Added CharTokenizer import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_gpt_parameters(d_model, n_layers, vocab_size, n_head, d_hid, max_seq_length):
    """
    Calculate expected parameter count for GPT-style model.
    
    Args:
        d_model: Embedding dimension
        n_layers: Number of transformer layers
        vocab_size: Size of vocabulary
        n_head: Number of attention heads
        d_hid: Hidden dimension in feed-forward network
        max_seq_length: Maximum sequence length (for positional embeddings)
        
    Returns:
        Expected parameter count
    """
    # Token embeddings: vocab_size * d_model
    token_emb_params = vocab_size * d_model
    
    # Position embeddings: max_seq_length * d_model
    pos_emb_params = max_seq_length * d_model
    
    # Self-attention per layer:
    # 3 projection matrices (Q, K, V): 3 * d_model * d_model
    # Output projection: d_model * d_model
    attn_params_per_layer = 4 * d_model * d_model
    
    # Feed-forward network per layer:
    # First projection: d_model * d_hid
    # Second projection: d_hid * d_model
    ffn_params_per_layer = d_model * d_hid + d_hid * d_model
    
    # Layer norm per layer (2 per layer): 2 * 2 * d_model
    # 2 params per neuron (weight and bias), 2 layer norms per transformer layer
    ln_params_per_layer = 4 * d_model
    
    # Parameters per layer
    params_per_layer = attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer
    
    # Total transformer layers
    transformer_params = params_per_layer * n_layers
    
    # Final layer norm
    final_ln_params = 2 * d_model
    
    # Output projection
    output_params = d_model * vocab_size
    
    # Biases (optional calculation)
    # Each linear layer has a bias vector
    biases = 0
    
    # Total parameters
    total_params = (
        token_emb_params +
        pos_emb_params +
        transformer_params +
        final_ln_params +
        output_params +
        biases
    )
    
    return {
        'token_embeddings': token_emb_params,
        'position_embeddings': pos_emb_params,
        'attention_layers': attn_params_per_layer * n_layers,
        'feedforward_layers': ffn_params_per_layer * n_layers,
        'layer_norms': ln_params_per_layer * n_layers + final_ln_params,
        'output_projection': output_params,
        'biases': biases,
        'total': total_params
    }


def analyze_model_params(config_path=None):
    """
    Analyze model parameters based on either config or code.
    
    Args:
        config_path: Path to model configuration file
    """
    # Load configuration if provided
    if config_path:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model parameters
        d_model = config.get('d_model', 768)
        n_layers = config.get('n_layers', 12)
        vocab_size = config.get('vocab_size', 95)  # Default in config
        n_head = config.get('n_head', 12)
        d_hid = config.get('d_hid', 3072)
        max_seq_length = config.get('n_positions', 1024)
    else:
        # Default GPT-2 small parameters
        d_model = 768
        n_layers = 12
        vocab_size = 95  # Default char vocab
        n_head = 12
        d_hid = 3072
        max_seq_length = 1024
    
    # Calculate expected parameters
    expected_params = calculate_gpt_parameters(
        d_model, n_layers, vocab_size, n_head, d_hid, max_seq_length
    )
    
    # Create actual model for comparison
    logger.info(f"Creating model with vocab_size={vocab_size}")
    model = create_transformer_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        d_hid=d_hid,
        n_layers=n_layers,
        max_seq_length=max_seq_length
    )
    
    # Calculate actual parameter count
    actual_params = sum(p.numel() for p in model.parameters())
    
    # Print comparison
    logger.info("\n" + "="*50)
    logger.info(f"MODEL PARAMETER ANALYSIS")
    logger.info("="*50)
    logger.info(f"Configuration:")
    logger.info(f"  d_model: {d_model}")
    logger.info(f"  n_layers: {n_layers}")
    logger.info(f"  vocab_size: {vocab_size}")
    logger.info(f"  n_head: {n_head}")
    logger.info(f"  d_hid: {d_hid}")
    logger.info(f"  max_seq_length: {max_seq_length}")
    logger.info("\nParameter Breakdown:")
    
    for component, count in expected_params.items():
        if component != 'total':
            logger.info(f"  {component}: {count:,} ({count/expected_params['total']*100:.1f}%)")
    
    logger.info("\nComparison:")
    logger.info(f"  Expected parameter count: {expected_params['total']:,}")
    logger.info(f"  Actual parameter count:   {actual_params:,}")
    logger.info(f"  Difference:               {actual_params - expected_params['total']:,}")
    
    # # Check for any potential issues (Example for old CharDataset/110M setup)
    # if vocab_size == 95 and actual_params > 114_000_000:
    #     logger.info("\nPotential issue detected:")
    #     logger.info("  CharDataset vocabulary size might be different from config.") 
    #     logger.info("  Try running with a sample text file to check actual vocabulary size.")
    
    return expected_params, actual_params


def check_dataset_vocab_size(data_path):
    """
    Check the actual vocabulary size of a dataset using TextDataset.
    
    Args:
        data_path: Path to the text data
    """
    logger.info(f"Checking vocabulary size in {data_path}")
    
    # Instantiate a simple CharTokenizer (adjust if needed)
    tokenizer = CharTokenizer()
    
    # Load the dataset using TextDataset (assuming a small block_size is fine for vocab check)
    # Note: This will read the whole file and tokenize it, which might be slow for large files.
    try:
        # Use a dummy block_size; we only care about the tokenizer's vocab built from the text
        dataset = TextDataset(file_paths=[data_path], block_size=10, tokenizer=tokenizer)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load dataset to check vocab: {e}", exc_info=True)
        return None

    # Print vocabulary size
    actual_vocab_size = dataset.vocab_size
    logger.info(f"Actual vocabulary size from TextDataset tokenizer: {actual_vocab_size}")
    if hasattr(tokenizer, 'char_to_idx'):
         logger.info(f"Characters in vocabulary: {sorted(tokenizer.char_to_idx.keys())}")
    
    return actual_vocab_size


def main():
    parser = argparse.ArgumentParser(description="Analyze model parameters")
    parser.add_argument("--config", type=str, default="conf/models/chatgot_small_char.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--data", type=str, default="data/got/game_of_thrones.txt",
                       help="Path to data file for checking vocabulary size")
    parser.add_argument("--check_data", action="store_true",
                       help="Check dataset vocabulary size")
    parser.add_argument("--vocab_size", type=int, default=None,
                       help="Override the vocabulary size for calculation")
    
    args = parser.parse_args()
    
    if args.check_data:
        vocab_size = check_dataset_vocab_size(args.data)
        if args.vocab_size is None:
            # Also run the parameter analysis with the detected vocab size
            logger.info(f"\nRunning parameter analysis with detected vocab size: {vocab_size}")
            config_path = args.config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['vocab_size'] = vocab_size
            analyze_model_params(config_path)
    else:
        if args.vocab_size is not None:
            # Override vocab size in config
            config_path = args.config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['vocab_size'] = args.vocab_size
            logger.info(f"Overriding vocabulary size to {args.vocab_size}")
        analyze_model_params(args.config)


if __name__ == "__main__":
    main() 