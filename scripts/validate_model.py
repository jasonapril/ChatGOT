#!/usr/bin/env python
"""
Validate the Craft model.

This script loads a trained model and evaluates it on a validation dataset,
calculating perplexity and other relevant metrics.
"""
import os
import sys
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import Craft modules
from craft.models.model import load_model
from craft.models.generate import generate_text
from craft.data.dataset import load_processed_data
from craft.utils.logging import get_logger
from craft.utils.metrics import calculate_perplexity, calculate_metrics

logger = get_logger(__name__)


def validate_model(args):
    """
    Validate a trained model on a validation dataset.
    
    Args:
        args: Command line arguments
    
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    logger.info(f"Loading validation data from {args.data_path}")
    val_data = load_processed_data(args.data_path)
    
    # Set the device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Evaluate the model
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, val_data, device, args.batch_size)
    
    # Log the metrics
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    
    # Generate sample outputs
    if args.generate_samples:
        logger.info("Generating sample outputs...")
        samples = generate_samples(model, val_data, args.num_samples, args.max_length, device)
        
        for i, (prompt, generated) in enumerate(samples):
            logger.info(f"Sample {i+1}:")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated}")
            logger.info("-" * 40)
    
    return metrics


def evaluate_model(model, data, device, batch_size=32) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        data: The dataset to evaluate on
        device: The device to run on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            logits = model(input_ids)
            loss = model.compute_loss(logits, target_ids)
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0) * input_ids.size(1)
    
    avg_loss = total_loss / len(data)
    perplexity = calculate_perplexity(avg_loss)
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity
    }
    
    # Calculate additional metrics if possible
    if hasattr(model, "get_metrics"):
        additional_metrics = model.get_metrics(data, device, batch_size)
        metrics.update(additional_metrics)
    
    return metrics


def generate_samples(model, data, num_samples=5, max_length=100, device="cpu") -> List[Tuple[str, str]]:
    """
    Generate sample outputs from the model.
    
    Args:
        model: The model to generate from
        data: The dataset to sample prompts from
        num_samples: Number of samples to generate
        max_length: Maximum length of generated text
        device: The device to run on
    
    Returns:
        List of (prompt, generated_text) tuples
    """
    model.eval()
    samples = []
    
    # Sample random prompts from the data
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    
    for idx in indices:
        sample = data[idx]
        input_ids = sample["input_ids"].to(device)
        
        # Use a shorter prompt
        prompt_length = min(20, input_ids.size(1))
        prompt_ids = input_ids[:, :prompt_length]
        
        # Generate text
        generated_ids = generate_text(
            model, 
            prompt_ids, 
            max_new_tokens=max_length, 
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        
        # Decode the text
        prompt_text = model.decode(prompt_ids[0].cpu().numpy())
        generated_text = model.decode(generated_ids[0].cpu().numpy())
        
        samples.append((prompt_text, generated_text))
    
    return samples


def main():
    """Parse command line arguments and run the validation."""
    parser = argparse.ArgumentParser(description="Validate Craft model")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Path to the trained model"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True, 
        help="Path to the validation data"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        choices=["cuda", "cpu"], 
        help="Device to run on"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--generate-samples", 
        action="store_true", 
        help="Generate sample outputs"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=5, 
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=100, 
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Run validation
    try:
        validate_model(args)
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        raise


if __name__ == "__main__":
    main() 