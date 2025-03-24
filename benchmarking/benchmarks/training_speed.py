"""
Training Speed Benchmark
=======================

This benchmark measures training speed in samples per second and memory usage
for the training process. It runs a small number of training iterations
and records throughput metrics.
"""

import logging
import os
import time
import torch
from typing import Dict, Any
import argparse
import json

from src.model import Model
from src.data_handler import DataHandler
from src.monitoring.throughput_core import ThroughputMonitor
from src.monitoring.instrumentation import create_instrumented_model, create_instrumented_dataloader
from src.training.training_loop import train_epoch

def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the training speed benchmark.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results
    """
    logging.info("Starting training speed benchmark")
    
    # Create small dataset for benchmark
    benchmark_data_dir = os.path.join(args.data_dir, "benchmark")
    os.makedirs(benchmark_data_dir, exist_ok=True)
    
    # Create a sample dataset if it doesn't exist
    train_file = os.path.join(benchmark_data_dir, "train.txt")
    val_file = os.path.join(benchmark_data_dir, "val.txt")
    vocab_file = os.path.join(benchmark_data_dir, "vocab.json")
    
    vocab_size = 1000
    seq_length = 64
    
    if not all(os.path.exists(f) for f in [train_file, val_file, vocab_file]):
        # Generate a synthetic dataset for benchmarking
        _generate_benchmark_dataset(benchmark_data_dir, train_file, val_file, vocab_file)
    
    # Load vocab
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
        vocab_size = len(vocab)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    # Create dataloaders
    def create_dataloader(file, batch_size):
        dataset = torch.load(file) if file.endswith('.pt') else _load_text_file(file, seq_length)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
    
    batch_sizes = [8, 16, 32, 64, 128]
    results = {
        "device": device.type,
        "batch_sizes": {},
    }
    
    # Test different batch sizes
    for batch_size in batch_sizes:
        logging.info(f"Testing batch size: {batch_size}")
        
        try:
            # Create fresh dataloader for each test
            dataloader = create_dataloader(train_file, batch_size)
            
            # Set up monitoring
            monitor = ThroughputMonitor()
            instrumented_model = create_instrumented_model(model, monitor)
            instrumented_dataloader = create_instrumented_dataloader(dataloader, monitor)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Warmup 
            _run_warmup(instrumented_model, instrumented_dataloader, optimizer, device)
            
            # Measure training speed
            num_iterations = min(20, len(dataloader))
            monitor.reset()
            
            # Run training
            train_epoch(
                model=instrumented_model,
                optimizer=optimizer,
                scheduler=None,
                dataloader=instrumented_dataloader,
                device=device,
                epoch=1,
                max_iterations=num_iterations,
                use_amp=False,
                gradient_accumulation_steps=1,
                gradient_clip_val=None
            )
            
            # Collect results
            metrics = monitor.get_summary()
            
            # Extract key metrics
            throughput = metrics["throughput"]["samples_per_second"]
            memory_usage = metrics["memory"]["allocated_gb"]
            peak_memory = metrics["memory"]["peak_gb"]
            component_breakdown = metrics["component_breakdown"]
            
            results["batch_sizes"][str(batch_size)] = {
                "throughput": throughput,
                "memory_allocated_gb": memory_usage,
                "memory_peak_gb": peak_memory,
                "component_breakdown": component_breakdown
            }
            
            logging.info(f"Batch size {batch_size}: {throughput:.2f} samples/sec, "
                         f"{memory_usage:.2f} GB memory")
            
            # Release memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except RuntimeError as e:
            # Handle out of memory errors
            logging.warning(f"Failed with batch size {batch_size}: {e}")
            results["batch_sizes"][str(batch_size)] = {"error": str(e)}
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Find optimal batch size
    best_batch_size = None
    best_throughput = 0
    
    for batch_size, metrics in results["batch_sizes"].items():
        if "throughput" in metrics and metrics["throughput"] > best_throughput:
            best_throughput = metrics["throughput"]
            best_batch_size = batch_size
    
    results["optimal_batch_size"] = best_batch_size
    results["peak_throughput"] = best_throughput
    
    return results

def _generate_benchmark_dataset(output_dir, train_file, val_file, vocab_file, 
                               dataset_size=10000, vocab_size=1000):
    """Generate a synthetic dataset for benchmarking."""
    logging.info("Generating synthetic benchmark dataset")
    
    import numpy as np
    import json
    
    # Create a vocabulary
    vocab = {"<pad>": 0}
    for i in range(1, vocab_size):
        vocab[f"token_{i}"] = i
    
    # Save vocabulary
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    
    # Generate synthetic data
    seq_length = 64
    
    # Create training data
    train_data = []
    for _ in range(dataset_size):
        # Generate random sequence
        seq = np.random.randint(1, vocab_size, size=seq_length)
        train_data.append(torch.tensor(seq, dtype=torch.long))
    
    # Create validation data (smaller)
    val_data = []
    for _ in range(dataset_size // 10):
        seq = np.random.randint(1, vocab_size, size=seq_length)
        val_data.append(torch.tensor(seq, dtype=torch.long))
    
    # Save as tensor datasets
    torch.save(train_data, train_file)
    torch.save(val_data, val_file)
    
    logging.info(f"Saved synthetic dataset to {output_dir}")

def _load_text_file(file_path, seq_length):
    """Load text file and convert to dataset of sequences."""
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Simple character tokenization for benchmark purposes
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}
    
    # Convert text to indices
    indices = [char_to_idx.get(ch, 0) for ch in text]
    
    # Create sequences
    sequences = []
    for i in range(0, len(indices) - seq_length, seq_length):
        seq = torch.tensor(indices[i:i+seq_length], dtype=torch.long)
        sequences.append(seq)
    
    return sequences

def _run_warmup(model, dataloader, optimizer, device, num_iterations=3):
    """Run a few warmup iterations to initialize CUDA."""
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= num_iterations:
            break
            
        # Move data to device
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step() 