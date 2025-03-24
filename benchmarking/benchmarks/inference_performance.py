"""
Inference Performance Benchmark
==============================

This benchmark measures inference performance metrics including:
- Generation speed (tokens per second)
- Memory usage during generation
- Latency for first token and subsequent tokens
- Scaling with batch size
"""

import logging
import os
import time
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import argparse
import json

from src.model import Model
from src.training.generation import batch_generate, generate_text
from src.monitoring.throughput_core import ThroughputMonitor

def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the inference performance benchmark.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results
    """
    logging.info("Starting inference performance benchmark")
    
    # Create or load model for benchmark
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_info = _prepare_model(args, device)
    model = model_info["model"]
    vocab_size = model_info["vocab_size"]
    char_to_idx = model_info.get("char_to_idx")
    idx_to_char = model_info.get("idx_to_char")
    
    # Run benchmark iterations
    results = {
        "device": device.type,
        "model_size": sum(p.numel() for p in model.parameters()),
        "vocab_size": vocab_size,
        "single_token_latency": {},
        "batch_scaling": {},
        "sequence_length_scaling": {}
    }
    
    # 1. Measure single token generation latency
    logging.info("Measuring single token generation latency")
    results["single_token_latency"] = _benchmark_single_token_latency(
        model, char_to_idx, idx_to_char, device
    )
    
    # 2. Measure batch scaling
    logging.info("Measuring batch size scaling")
    results["batch_scaling"] = _benchmark_batch_scaling(
        model, char_to_idx, idx_to_char, device
    )
    
    # 3. Measure sequence length scaling
    logging.info("Measuring sequence length scaling")
    results["sequence_length_scaling"] = _benchmark_sequence_length_scaling(
        model, char_to_idx, idx_to_char, device
    )
    
    # Calculate overall metrics
    results["avg_tokens_per_second"] = results["single_token_latency"]["tokens_per_second"]
    results["memory_usage_gb"] = results["single_token_latency"]["memory_usage_gb"]
    
    # Find optimal batch size
    batch_sizes = results["batch_scaling"].keys()
    if batch_sizes:
        best_batch_size = max(
            batch_sizes, 
            key=lambda b: results["batch_scaling"][b].get("tokens_per_second", 0)
        )
        results["optimal_batch_size"] = best_batch_size
        results["optimal_throughput"] = results["batch_scaling"][best_batch_size]["tokens_per_second"]
    
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
        # Create a new model for benchmarking
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

def _benchmark_single_token_latency(
    model, char_to_idx, idx_to_char, device, num_runs=10
) -> Dict[str, float]:
    """Benchmark latency for single token generation."""
    
    # Prepare inputs
    seed_text = "The quick brown fox jumps over the lazy dog"
    if char_to_idx:
        # Convert to tensor
        input_tokens = torch.tensor(
            [char_to_idx.get(c, 0) for c in seed_text], 
            dtype=torch.long
        ).to(device)
    else:
        # Just use some random tokens
        input_tokens = torch.randint(0, model.vocab_size, (len(seed_text),)).to(device)
    
    input_tokens = input_tokens.unsqueeze(0)  # Add batch dimension
    
    # Warmup
    with torch.no_grad():
        _ = model(input_tokens)
    
    # Clear memory 
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Record memory before generation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    else:
        start_memory = 0
    
    # Measure first token latency
    first_token_times = []
    for _ in range(num_runs):
        with torch.no_grad():
            start_time = time.time()
            _ = model(input_tokens)
            first_token_times.append(time.time() - start_time)
    
    # Measure per-token latency for subsequent tokens
    subsequent_token_times = []
    for _ in range(num_runs):
        with torch.no_grad():
            # Generate 20 subsequent tokens
            start_time = time.time()
            current_input = input_tokens
            
            for _ in range(20):
                output = model(current_input)
                # Get last token
                next_token = torch.argmax(output[:, -1, :], dim=1, keepdim=True)
                # Append to input
                current_input = torch.cat([current_input, next_token], dim=1)
                
            generation_time = time.time() - start_time
            tokens_generated = 20
            subsequent_token_times.append(generation_time / tokens_generated)
    
    # Record memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        memory_usage = peak_memory - start_memory
    else:
        memory_usage = 0
    
    # Calculate metrics
    avg_first_token = np.mean(first_token_times) * 1000  # ms
    avg_subsequent_token = np.mean(subsequent_token_times) * 1000  # ms
    tokens_per_second = 1.0 / (avg_subsequent_token / 1000)
    
    return {
        "first_token_latency_ms": avg_first_token,
        "subsequent_token_latency_ms": avg_subsequent_token,
        "tokens_per_second": tokens_per_second,
        "memory_usage_gb": memory_usage
    }

def _benchmark_batch_scaling(model, char_to_idx, idx_to_char, device) -> Dict[str, Dict[str, float]]:
    """Benchmark how performance scales with batch size."""
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = {}
    
    seed_text = "The quick brown fox jumps over the lazy dog"
    if char_to_idx:
        input_ids = [char_to_idx.get(c, 0) for c in seed_text]
    else:
        input_ids = list(range(min(len(seed_text), 10)))
    
    for batch_size in batch_sizes:
        logging.info(f"Testing batch size: {batch_size}")
        
        try:
            # Create batch of prompts
            prompts = [seed_text] * batch_size
            
            # Warmup
            with torch.no_grad():
                if char_to_idx and idx_to_char:
                    _ = batch_generate(
                        model, prompts, char_to_idx, idx_to_char, max_length=10, 
                        temperature=1.0, device=device
                    )
                else:
                    # Just measure raw model forward pass
                    input_tensor = torch.tensor([input_ids] * batch_size, dtype=torch.long).to(device)
                    _ = model(input_tensor)
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Measure generation time
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            else:
                start_memory = 0
                
            generation_length = 50
            start_time = time.time()
            
            with torch.no_grad():
                if char_to_idx and idx_to_char:
                    _ = batch_generate(
                        model, prompts, char_to_idx, idx_to_char, 
                        max_length=generation_length, temperature=1.0, device=device
                    )
                else:
                    # Simulate generation
                    input_tensor = torch.tensor([input_ids] * batch_size, dtype=torch.long).to(device)
                    for _ in range(generation_length):
                        output = model(input_tensor)
                        next_tokens = torch.argmax(output[:, -1, :], dim=1, keepdim=True)
                        input_tensor = torch.cat([input_tensor, next_tokens], dim=1)
            
            generation_time = time.time() - start_time
            
            # Record memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                memory_usage = peak_memory - start_memory
            else:
                memory_usage = 0
            
            # Calculate metrics
            tokens_per_second = (batch_size * generation_length) / generation_time
            effective_tokens_per_second = tokens_per_second
            
            results[str(batch_size)] = {
                "tokens_per_second": tokens_per_second,
                "effective_tokens_per_second": effective_tokens_per_second,
                "memory_usage_gb": memory_usage,
                "throughput_efficiency": effective_tokens_per_second / (tokens_per_second / batch_size)
            }
            
            logging.info(f"Batch {batch_size}: {tokens_per_second:.2f} tokens/sec, {memory_usage:.2f} GB")
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except RuntimeError as e:
            logging.warning(f"Failed with batch size {batch_size}: {e}")
            results[str(batch_size)] = {"error": str(e)}
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def _benchmark_sequence_length_scaling(
    model, char_to_idx, idx_to_char, device
) -> Dict[str, Dict[str, float]]:
    """Benchmark how performance scales with sequence length."""
    seq_lengths = [32, 64, 128, 256, 512, 1024]
    results = {}
    
    seed_text = "The quick brown fox jumps over the lazy dog"
    if char_to_idx:
        # Convert to tokens
        tokens = [char_to_idx.get(c, 0) for c in seed_text]
    else:
        # Use random tokens
        tokens = list(range(min(len(seed_text), 10)))
    
    for seq_length in seq_lengths:
        logging.info(f"Testing sequence length: {seq_length}")
        
        try:
            # Create input sequence of desired length by repeating tokens
            input_seq = tokens * (seq_length // len(tokens) + 1)
            input_seq = input_seq[:seq_length]
            
            # Convert to tensor
            input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
            
            # Warmup
            with torch.no_grad():
                _ = model(input_tensor)
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Measure forward pass time
            num_runs = 10
            forward_times = []
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated() / (1024 ** 3)
            else:
                start_memory = 0
                
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    _ = model(input_tensor)
                forward_times.append(time.time() - start_time)
            
            # Record memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                memory_usage = peak_memory - start_memory
            else:
                memory_usage = 0
            
            # Calculate metrics
            avg_forward_time = np.mean(forward_times)
            tokens_per_second = seq_length / avg_forward_time
            
            results[str(seq_length)] = {
                "forward_time_ms": avg_forward_time * 1000,
                "tokens_per_second": tokens_per_second,
                "memory_usage_gb": memory_usage
            }
            
            logging.info(f"Seq length {seq_length}: {tokens_per_second:.2f} tokens/sec, {memory_usage:.2f} GB")
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except RuntimeError as e:
            logging.warning(f"Failed with sequence length {seq_length}: {e}")
            results[str(seq_length)] = {"error": str(e)}
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results 