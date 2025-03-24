"""
Model Accuracy Benchmark
======================

This benchmark evaluates model accuracy metrics including:
- Perplexity on validation data
- Error rate at different temperatures
- Sample quality metrics
- Token prediction accuracy
"""

import logging
import os
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import argparse
import json
import math
import re

from src.model import Model
from src.training.evaluation import evaluate_perplexity
from src.training.generation import generate_text, sample_text

def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the model accuracy benchmark.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results
    """
    logging.info("Starting model accuracy benchmark")
    
    # Create or load model for benchmark
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not args.model_checkpoint:
        logging.warning("No model checkpoint provided, accuracy metrics will be meaningless")
    
    model_info = _prepare_model(args, device)
    model = model_info["model"]
    vocab_size = model_info["vocab_size"]
    char_to_idx = model_info.get("char_to_idx")
    idx_to_char = model_info.get("idx_to_char")
    
    # Initialize results
    results = {
        "device": device.type,
        "model_size": sum(p.numel() for p in model.parameters()),
        "vocab_size": vocab_size,
        "perplexity": {},
        "generation_quality": {},
        "token_prediction": {}
    }
    
    # 1. Measure perplexity on test data if available
    test_file = _find_test_file(args)
    if test_file:
        logging.info(f"Measuring perplexity using test file: {test_file}")
        results["perplexity"] = _benchmark_perplexity(
            model, test_file, char_to_idx, device
        )
    
    # 2. Evaluate generation quality at different temperatures
    logging.info("Evaluating generation quality at different temperatures")
    results["generation_quality"] = _benchmark_generation_quality(
        model, char_to_idx, idx_to_char, device
    )
    
    # 3. Evaluate token prediction accuracy
    logging.info("Evaluating token prediction accuracy")
    results["token_prediction"] = _benchmark_token_prediction(
        model, test_file, char_to_idx, idx_to_char, device
    )
    
    # Calculate aggregate metrics
    if "validation_perplexity" in results["perplexity"]:
        results["validation_perplexity"] = results["perplexity"]["validation_perplexity"]
    
    if "avg_repetition_score" in results["generation_quality"]:
        results["repetition_score"] = results["generation_quality"]["avg_repetition_score"]
    
    if "top1_accuracy" in results["token_prediction"]:
        results["token_prediction_accuracy"] = results["token_prediction"]["top1_accuracy"]
    
    return results

def _prepare_model(args, device) -> Dict[str, Any]:
    """Set up model for benchmarking."""
    # Check if we have a checkpoint to load
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        # Load model from checkpoint
        logging.info(f"Loading model from checkpoint: {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        
        # Load vocab
        vocab_file = None
        for ext in ['.json', '.txt', '.vocab']:
            potential_vocab = os.path.join(os.path.dirname(args.model_checkpoint), f"vocab{ext}")
            if os.path.exists(potential_vocab):
                vocab_file = potential_vocab
                break
                
        if vocab_file:
            logging.info(f"Loading vocabulary from {vocab_file}")
            if vocab_file.endswith('.json'):
                with open(vocab_file, 'r') as f:
                    vocab = json.load(f)
                    char_to_idx = vocab
                    idx_to_char = {idx: char for char, idx in vocab.items()}
                    vocab_size = len(vocab)
            else:
                # Simple text file with one token per line
                with open(vocab_file, 'r') as f:
                    chars = [line.strip() for line in f]
                    char_to_idx = {ch: i for i, ch in enumerate(chars)}
                    idx_to_char = {i: ch for i, ch in enumerate(chars)}
                    vocab_size = len(chars)
        else:
            # No vocab file found, use checkpoint info if available
            vocab_size = checkpoint.get("vocab_size", 1000)
            char_to_idx = None
            idx_to_char = None
            logging.warning("No vocabulary file found, using checkpoint info")
        
        # Create model with same architecture
        model = Model(
            vocab_size=vocab_size,
            embedding_dim=checkpoint.get("embedding_dim", 128),
            hidden_size=checkpoint.get("hidden_size", 256),
            num_layers=checkpoint.get("num_layers", 2),
            dropout=0.0  # No dropout for inference
        ).to(device)
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
    else:
        # Create a new model for benchmarking (will have random weights)
        logging.info("Creating new model for benchmarking")
        vocab_size = 1000
        
        # Create synthetic vocab
        char_to_idx = {"<pad>": 0}
        for i in range(1, vocab_size):
            char_to_idx[f"t{i}"] = i
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        model = Model(
            vocab_size=vocab_size,
            embedding_dim=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.0
        ).to(device)
        model.eval()
    
    return {
        "model": model,
        "vocab_size": vocab_size,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char
    }

def _find_test_file(args) -> str:
    """Find appropriate test file for evaluation."""
    # Check for test file in checkpoint directory
    if args.model_checkpoint:
        checkpoint_dir = os.path.dirname(args.model_checkpoint)
        for name in ['test.txt', 'test.pt', 'val.txt', 'val.pt']:
            test_path = os.path.join(checkpoint_dir, name)
            if os.path.exists(test_path):
                return test_path
    
    # Check for test file in data directory
    data_dir = args.data_dir
    for name in ['test.txt', 'test.pt', 'val.txt', 'val.pt']:
        test_path = os.path.join(data_dir, name)
        if os.path.exists(test_path):
            return test_path
    
    # Create a small synthetic test file
    test_path = os.path.join(args.data_dir, "synthetic_test.txt")
    with open(test_path, "w") as f:
        f.write("This is a small synthetic test file created for benchmarking purposes. " +
                "It contains some basic text that can be used to evaluate the model's " +
                "perplexity and token prediction accuracy, though these metrics will " +
                "not be very meaningful with this synthetic data.")
    
    return test_path

def _benchmark_perplexity(model, test_file, char_to_idx, device) -> Dict[str, float]:
    """Benchmark model perplexity on test data."""
    # Load test data
    if not test_file or not os.path.exists(test_file):
        return {"error": "No test file found"}
    
    try:
        # Use evaluation function
        if not char_to_idx:
            return {"error": "No character to index mapping available"}
        
        # Create dataloader from test file
        if test_file.endswith('.pt'):
            # Pre-processed data
            test_data = torch.load(test_file)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
        else:
            # Text file - needs processing
            with open(test_file, 'r') as f:
                text = f.read()
            
            # Convert to sequence of tokens
            seq_length = 64  # Use a reasonable default
            indices = [char_to_idx.get(ch, 0) for ch in text]
            
            # Create sequences
            sequences = []
            for i in range(0, len(indices) - seq_length, seq_length):
                input_seq = torch.tensor(indices[i:i+seq_length], dtype=torch.long)
                sequences.append(input_seq)
            
            if not sequences:
                return {"error": "Test file too small"}
                
            test_loader = torch.utils.data.DataLoader(sequences, batch_size=16)
        
        # Measure perplexity
        avg_loss, perplexity = evaluate_perplexity(
            model=model,
            dataloader=test_loader,
            device=device
        )
        
        return {
            "validation_loss": avg_loss,
            "validation_perplexity": perplexity,
            "test_file": test_file
        }
        
    except Exception as e:
        logging.error(f"Error calculating perplexity: {e}")
        return {"error": str(e)}

def _benchmark_generation_quality(
    model, char_to_idx, idx_to_char, device
) -> Dict[str, float]:
    """Benchmark generation quality metrics."""
    if not char_to_idx or not idx_to_char:
        return {"error": "No character to index mapping available"}
    
    try:
        # Generate samples at different temperatures
        temperatures = [0.5, 0.7, 1.0, 1.3]
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In the beginning",
            "It was a dark and stormy night"
        ]
        
        results = {
            "temperature_samples": {},
            "avg_repetition_score": 0,
            "avg_entropy": 0
        }
        
        all_repetition_scores = []
        all_entropy_scores = []
        
        for temp in temperatures:
            logging.info(f"Generating samples at temperature {temp}")
            temp_results = []
            
            for prompt in prompts:
                # Generate text
                generated = generate_text(
                    model=model,
                    char_to_idx=char_to_idx,
                    idx_to_char=idx_to_char,
                    seed_text=prompt,
                    max_length=200,
                    temperature=temp,
                    device=device
                )
                
                # Calculate metrics
                repetition_score = _calculate_repetition_score(generated)
                entropy = _calculate_entropy(generated)
                
                all_repetition_scores.append(repetition_score)
                all_entropy_scores.append(entropy)
                
                temp_results.append({
                    "prompt": prompt,
                    "generated_text": generated,
                    "repetition_score": repetition_score,
                    "entropy": entropy
                })
            
            # Add to results
            results["temperature_samples"][str(temp)] = temp_results
        
        # Calculate averages
        results["avg_repetition_score"] = np.mean(all_repetition_scores)
        results["avg_entropy"] = np.mean(all_entropy_scores)
        
        return results
        
    except Exception as e:
        logging.error(f"Error evaluating generation quality: {e}")
        return {"error": str(e)}

def _benchmark_token_prediction(
    model, test_file, char_to_idx, idx_to_char, device
) -> Dict[str, float]:
    """Benchmark token prediction accuracy."""
    if not test_file or not os.path.exists(test_file) or not char_to_idx:
        return {"error": "Missing test file or character mapping"}
    
    try:
        # Load a sample of text from test file
        with open(test_file, 'r') as f:
            text = f.read()
        
        # Limit text size for prediction
        max_text_size = 10000
        if len(text) > max_text_size:
            text = text[:max_text_size]
        
        # Convert to sequence of tokens
        seq_length = 64  # Context length
        indices = [char_to_idx.get(ch, 0) for ch in text]
        
        # Skip text that's too short
        if len(indices) < seq_length + 100:
            return {"error": "Test file too small for token prediction"}
        
        # Prepare for prediction
        correct_top1 = 0
        correct_top5 = 0
        total_predictions = 0
        
        # Sample positions for prediction
        step = max(1, len(indices) // 100)  # Aim for ~100 samples
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(indices) - seq_length - 1, step):
                # Get context and target
                context = indices[i:i+seq_length]
                target = indices[i+seq_length]
                
                # Convert to tensor
                context_tensor = torch.tensor([context], dtype=torch.long).to(device)
                
                # Get prediction
                output = model(context_tensor)
                
                # Get top predictions
                logits = output[0, -1, :]
                top_values, top_indices = torch.topk(logits, 5)
                
                # Check for correct predictions
                if target == top_indices[0].item():
                    correct_top1 += 1
                    correct_top5 += 1
                elif target in top_indices.tolist():
                    correct_top5 += 1
                
                total_predictions += 1
        
        # Calculate accuracy
        top1_accuracy = correct_top1 / total_predictions if total_predictions > 0 else 0
        top5_accuracy = correct_top5 / total_predictions if total_predictions > 0 else 0
        
        return {
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "total_samples": total_predictions
        }
        
    except Exception as e:
        logging.error(f"Error evaluating token prediction: {e}")
        return {"error": str(e)}

def _calculate_repetition_score(text: str) -> float:
    """
    Calculate a repetition score based on repeated n-grams.
    Lower score is better (less repetition).
    """
    if not text or len(text) < 10:
        return 0.0
    
    # Normalize text
    text = text.lower()
    
    # Check for repetition at different n-gram sizes
    repetition_scores = []
    
    for n in [3, 4, 5]:
        # Skip if text is too short
        if len(text) < n*2:
            continue
            
        # Create n-grams
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        
        # Count occurrence of each n-gram
        counts = {}
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1
        
        # Calculate repetition score (average repeat count - 1)
        repeats = [count-1 for count in counts.values() if count > 1]
        
        if repeats:
            repetition_scores.append(sum(repeats) / len(ngrams))
        else:
            repetition_scores.append(0.0)
    
    # Average across n-gram sizes
    if repetition_scores:
        return sum(repetition_scores) / len(repetition_scores)
    return 0.0

def _calculate_entropy(text: str) -> float:
    """Calculate character-level entropy of the text."""
    if not text:
        return 0.0
    
    # Count character frequencies
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Calculate entropy
    length = len(text)
    entropy = 0
    
    for count in char_counts.values():
        prob = count / length
        entropy -= prob * math.log2(prob)
    
    return entropy 