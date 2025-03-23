#!/usr/bin/env python
"""
Benchmark Script for PyTorch 2.0+ Compilation
=============================================

This script measures the performance impact of torch.compile on a real dataset.
It runs training with and without compilation and reports throughput metrics.
"""

import argparse
import torch
import time
import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model import create_transformer_model, TransformerModel
from src.trainer import train_epoch
from src.data_handler import load_data
from src.utils import setup_device, set_seed
from src.logger import setup_logger, log_section_header

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark torch.compile performance on real training data."
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the processed data file.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Maximum sequence length.")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model embedding dimension.")
    parser.add_argument("--n_head", type=int, default=8,
                        help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of transformer layers.")
    parser.add_argument("--d_hid", type=int, default=1024,
                        help="Hidden dimension of feedforward layers.")
    
    # Benchmark arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs to benchmark.")
    parser.add_argument("--warmup_epochs", type=int, default=1,
                        help="Number of warmup epochs before benchmarking.")
    parser.add_argument("--compile_modes", type=str, nargs='+', 
                        default=["default", "reduce-overhead", "max-autotune"],
                        help="Compilation modes to benchmark.")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations to run for each configuration.")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results.")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Force using CPU even if CUDA is available.")
    
    return parser.parse_args()

def run_benchmark(model, optimizer, scheduler, dataloader, args, device, 
                 use_compile=False, compile_mode=None, warmup_epochs=0, benchmark_epochs=1, 
                 name="baseline"):
    """
    Run a benchmark with specified configuration.
    
    Args:
        model: The model to benchmark
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        dataloader: Training data loader
        args: Command line arguments
        device: Device to run on
        use_compile: Whether to use torch.compile
        compile_mode: Compilation mode for torch.compile
        warmup_epochs: Number of warmup epochs
        benchmark_epochs: Number of epochs to benchmark
        name: Name of this benchmark configuration
        
    Returns:
        dict: Benchmark results
    """
    logging.info(f"Running benchmark: {name}")
    
    # Clone model to ensure fair comparison
    model_copy = create_transformer_model(
        vocab_size=model.embedding.weight.size(0),
        max_seq_length=args.sequence_length,
        d_model=args.d_model,
        n_head=args.n_head,
        d_hid=args.d_hid,
        n_layers=args.n_layers,
        dropout=0.1
    )
    
    # Copy weights from original model
    model_copy.load_state_dict(model.state_dict())
    model_copy.to(device)
    
    # Create fresh optimizer and scheduler
    optimizer_copy = torch.optim.AdamW(model_copy.parameters(), lr=5e-5)
    scheduler_copy = torch.optim.lr_scheduler.LambdaLR(
        optimizer_copy, lambda _: 1.0
    )
    
    # Run warmup epochs
    for epoch in range(warmup_epochs):
        logging.info(f"Running warmup epoch {epoch+1}/{warmup_epochs}")
        _, _ = train_epoch(
            model=model_copy,
            optimizer=optimizer_copy,
            scheduler=scheduler_copy,
            dataloader=dataloader,
            device=device,
            epoch=epoch,
            use_torch_compile=use_compile,
            compile_mode=compile_mode
        )
    
    # Clear CUDA cache between runs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Run benchmark epochs
    benchmark_start_time = time.time()
    tokens_per_second = []
    train_losses = []
    
    for epoch in range(benchmark_epochs):
        logging.info(f"Running benchmark epoch {epoch+1}/{benchmark_epochs}")
        
        # Run a training epoch
        loss, tps = train_epoch(
            model=model_copy,
            optimizer=optimizer_copy,
            scheduler=scheduler_copy,
            dataloader=dataloader,
            device=device,
            epoch=epoch,
            use_torch_compile=use_compile,
            compile_mode=compile_mode
        )
        
        train_losses.append(loss)
        tokens_per_second.append(tps)
    
    benchmark_time = time.time() - benchmark_start_time
    
    # Calculate metrics
    avg_tokens_per_second = np.mean(tokens_per_second)
    peak_tokens_per_second = np.max(tokens_per_second)
    final_loss = train_losses[-1]
    
    logging.info(f"Benchmark {name} completed:")
    logging.info(f"- Average tokens/sec: {avg_tokens_per_second:.2f}")
    logging.info(f"- Peak tokens/sec: {peak_tokens_per_second:.2f}")
    logging.info(f"- Total time: {benchmark_time:.2f}s")
    logging.info(f"- Final loss: {final_loss:.6f}")
    
    return {
        'name': name,
        'avg_tokens_per_second': avg_tokens_per_second,
        'peak_tokens_per_second': peak_tokens_per_second,
        'total_time': benchmark_time,
        'train_losses': train_losses,
        'tokens_per_second': tokens_per_second,
        'use_compile': use_compile,
        'compile_mode': compile_mode
    }

def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "benchmark.log")
    setup_logger(log_file, "INFO")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up device
    device, is_cuda, _ = setup_device(args.force_cpu)
    
    # Log system information
    log_section_header("SYSTEM INFORMATION")
    logging.info(f"Device: {device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        logging.info(f"PyTorch CUDA: {torch.version.cuda}")
        logging.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    logging.info(f"PyTorch Version: {torch.__version__}")
    
    # Check if torch.compile is available
    has_compile = hasattr(torch, 'compile')
    logging.info(f"torch.compile available: {has_compile}")
    if not has_compile:
        logging.warning("torch.compile is not available in this PyTorch version!")
        logging.info("Continuing with baseline benchmark only...")
    
    # Load data
    log_section_header("LOADING DATA")
    train_loader, _, char_to_idx, _ = load_data(
        args.data_path,
        batch_size=args.batch_size,
        device_type=device.type,
        num_workers=4
    )
    
    vocab_size = len(char_to_idx)
    logging.info(f"Vocabulary size: {vocab_size}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Sequence length: {args.sequence_length}")
    
    # Create model
    log_section_header("CREATING MODEL")
    model = create_transformer_model(
        vocab_size=vocab_size,
        max_seq_length=args.sequence_length,
        d_model=args.d_model,
        n_head=args.n_head,
        d_hid=args.d_hid,
        n_layers=args.n_layers,
        dropout=0.1
    )
    model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")
    logging.info(f"Model architecture:")
    logging.info(f"- d_model: {args.d_model}")
    logging.info(f"- n_head: {args.n_head}")
    logging.info(f"- n_layers: {args.n_layers}")
    logging.info(f"- d_hid: {args.d_hid}")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1.0
    )
    
    # Run benchmarks
    log_section_header("RUNNING BENCHMARKS")
    
    # Configure benchmarks to run
    benchmarks = [
        {
            'name': 'baseline',
            'use_compile': False,
            'compile_mode': None
        }
    ]
    
    if has_compile:
        for mode in args.compile_modes:
            benchmarks.append({
                'name': f'torch_compile_{mode}',
                'use_compile': True,
                'compile_mode': mode
            })
    
    all_results = []
    
    # Run each benchmark multiple times
    for iteration in range(args.iterations):
        log_section_header(f"ITERATION {iteration+1}/{args.iterations}")
        iteration_results = []
        
        for benchmark in benchmarks:
            # Run benchmark
            result = run_benchmark(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=train_loader,
                args=args,
                device=device,
                use_compile=benchmark['use_compile'],
                compile_mode=benchmark['compile_mode'],
                warmup_epochs=args.warmup_epochs,
                benchmark_epochs=args.epochs,
                name=f"{benchmark['name']}_iter{iteration+1}"
            )
            
            iteration_results.append(result)
        
        all_results.extend(iteration_results)
    
    # Calculate average results
    log_section_header("BENCHMARK RESULTS")
    
    # Organize results by benchmark configuration
    grouped_results = {}
    for result in all_results:
        name = result['name'].split('_iter')[0]
        if name not in grouped_results:
            grouped_results[name] = []
        grouped_results[name].append(result)
    
    # Calculate averages and print summary
    summary = []
    for name, results in grouped_results.items():
        avg_tps = np.mean([r['avg_tokens_per_second'] for r in results])
        peak_tps = np.max([r['peak_tokens_per_second'] for r in results])
        avg_time = np.mean([r['total_time'] for r in results])
        
        summary.append({
            'name': name,
            'avg_tokens_per_second': avg_tps,
            'peak_tokens_per_second': peak_tps,
            'avg_time': avg_time
        })
        
        logging.info(f"Configuration: {name}")
        logging.info(f"- Average tokens/sec: {avg_tps:.2f}")
        logging.info(f"- Peak tokens/sec: {peak_tps:.2f}")
        logging.info(f"- Average time: {avg_time:.2f}s")
    
    # Find the best configuration
    best_config = max(summary, key=lambda x: x['avg_tokens_per_second'])
    baseline = next(s for s in summary if s['name'] == 'baseline')
    
    logging.info("\nBest configuration:")
    logging.info(f"- {best_config['name']}: {best_config['avg_tokens_per_second']:.2f} tokens/sec")
    
    if best_config['name'] != 'baseline':
        speedup = best_config['avg_tokens_per_second'] / baseline['avg_tokens_per_second']
        logging.info(f"- Speedup vs baseline: {speedup:.2f}x")
    
    # Save detailed results
    import json
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    
    # Convert numpy values to native Python types for JSON serialization
    for result in all_results:
        result['avg_tokens_per_second'] = float(result['avg_tokens_per_second'])
        result['peak_tokens_per_second'] = float(result['peak_tokens_per_second'])
        result['total_time'] = float(result['total_time'])
        result['train_losses'] = [float(x) for x in result['train_losses']]
        result['tokens_per_second'] = [float(x) for x in result['tokens_per_second']]
    
    with open(results_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': all_results,
            'summary': summary
        }, f, indent=2)
    
    logging.info(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main() 