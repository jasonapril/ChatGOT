#!/usr/bin/env python
"""
Model Validation Script for ChatGoT

This script is used to validate trained models by:
1. Loading a trained model
2. Running validation metrics on the validation dataset
3. Generating sample text and evaluating quality
4. Testing inference performance

Usage:
    python validate_model.py --checkpoint <path_to_checkpoint>
    python validate_model.py --dir <directory_with_checkpoints>
    python validate_model.py --latest # Use latest checkpoint
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
try:
    from chatgot.models.model import load_model
    from chatgot.models.generate import generate_text
    from chatgot.data.dataset import load_processed_data
    from chatgot.utils.logging import get_logger
    from chatgot.utils.metrics import calculate_perplexity, calculate_metrics
except ImportError as e:
    print(f"Error importing ChatGoT modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

# Configure logging
logger = get_logger("validate_model")

def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint in the specified directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory doesn't exist: {checkpoint_dir}")
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)

def load_validation_data():
    """Load validation data for model evaluation."""
    try:
        data = load_processed_data()
        return data['val_sequences'], data['char_to_idx'], data['idx_to_char']
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        raise

def evaluate_model(model, val_data, char_to_idx, idx_to_char):
    """Evaluate the model on validation data."""
    results = {}
    
    # Measure inference time
    start_time = time.time()
    
    # Calculate perplexity
    logger.info("Calculating perplexity...")
    perplexity = calculate_perplexity(model, val_data)
    results['perplexity'] = perplexity
    
    # Calculate additional metrics
    metrics = calculate_metrics(model, val_data)
    results.update(metrics)
    
    # Generate text samples
    logger.info("Generating text samples...")
    samples = []
    prompts = [
        "TYRION: ",
        "JON SNOW: ",
        "DAENERYS: "
    ]
    
    for prompt in prompts:
        sample = generate_text(
            model=model,
            prompt=prompt,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            max_length=200,
            temperature=0.8
        )
        samples.append({
            'prompt': prompt,
            'generated': sample
        })
    
    results['samples'] = samples
    
    # Add inference time
    inference_time = time.time() - start_time
    results['inference_time'] = inference_time
    
    return results

def save_validation_results(results, checkpoint_path):
    """Save validation results to a file."""
    checkpoint_name = os.path.basename(checkpoint_path)
    validation_dir = os.path.join("validation_results")
    os.makedirs(validation_dir, exist_ok=True)
    
    output_path = os.path.join(validation_dir, f"{checkpoint_name}_validation.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation results saved to {output_path}")

def print_validation_summary(results):
    """Print a summary of validation results."""
    print("\n" + "=" * 70)
    print("MODEL VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"Perplexity: {results.get('perplexity', 'N/A'):.4f}")
    
    if 'loss' in results:
        print(f"Validation Loss: {results['loss']:.4f}")
    
    if 'accuracy' in results:
        print(f"Character Accuracy: {results['accuracy']*100:.2f}%")
    
    print(f"\nInference Time: {results.get('inference_time', 0):.2f} seconds")
    
    print("\nSAMPLE GENERATIONS:")
    for i, sample in enumerate(results.get('samples', [])):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {sample['prompt']}")
        print(f"Generated: {sample['generated']}")
    
    print("\n" + "=" * 70)

def main():
    """Run model validation."""
    parser = argparse.ArgumentParser(description="Validate ChatGoT model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    group.add_argument('--dir', type=str, help='Directory with checkpoints')
    group.add_argument('--latest', action='store_true', help='Use latest checkpoint')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--save', action='store_true', help='Save validation results')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    
    try:
        # Determine checkpoint path
        checkpoint_path = None
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif args.dir:
            checkpoint_path = find_latest_checkpoint(args.dir)
        elif args.latest:
            checkpoint_path = find_latest_checkpoint('checkpoints')
        
        logger.info(f"Using checkpoint: {checkpoint_path}")
        
        # Load model
        logger.info("Loading model...")
        model = load_model(checkpoint_path)
        
        # Load validation data
        logger.info("Loading validation data...")
        val_data, char_to_idx, idx_to_char = load_validation_data()
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = evaluate_model(model, val_data, char_to_idx, idx_to_char)
        
        # Save results if requested
        if args.save:
            save_validation_results(results, checkpoint_path)
        
        # Print summary
        print_validation_summary(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error validating model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 