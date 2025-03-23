#!/usr/bin/env python
"""
Benchmark Logger
===============

Handles logging of benchmark results for different optimization techniques.
"""

import os
import json
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

class BenchmarkLogger:
    """
    Logger for tracking and recording benchmark results.
    
    This class provides functionality to record benchmark results,
    compare different optimization techniques, and save results to disk.
    """
    
    def __init__(self, log_dir: str = "benchmark_logs"):
        """
        Initialize the benchmark logger.
        
        Args:
            log_dir: Directory to save benchmark logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.benchmarks = []
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_dir / "benchmarks.log")
            ]
        )
        
    def log_benchmark(self, 
                      name: str, 
                      throughput: float, 
                      batch_size: int,
                      model_config: Dict[str, Any],
                      optimizations: Dict[str, bool],
                      memory_usage_mb: Optional[float] = None,
                      additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a benchmark result.
        
        Args:
            name: Name of the benchmark
            throughput: Throughput in tokens/sec
            batch_size: Batch size used
            model_config: Dictionary containing model configuration parameters
            optimizations: Dictionary of optimization techniques used
            memory_usage_mb: Peak memory usage in MB (if available)
            additional_info: Any additional information to log
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        benchmark = {
            "name": name,
            "timestamp": timestamp,
            "throughput": throughput,
            "batch_size": batch_size,
            "model_config": model_config,
            "optimizations": optimizations,
            "memory_usage_mb": memory_usage_mb,
            "additional_info": additional_info or {}
        }
        
        self.benchmarks.append(benchmark)
        
        # Log to console and file
        logging.info(f"Benchmark: {name}")
        logging.info(f"  Throughput: {throughput:.2f} tokens/sec")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Model: d_model={model_config.get('d_model', 'N/A')}, n_layers={model_config.get('n_layers', 'N/A')}")
        
        if memory_usage_mb:
            logging.info(f"  Memory usage: {memory_usage_mb:.2f} MB")
        
        # Log optimizations
        opt_str = ", ".join([k for k, v in optimizations.items() if v])
        if opt_str:
            logging.info(f"  Optimizations: {opt_str}")
        else:
            logging.info("  Optimizations: None")
        
        # Save the updated benchmarks to disk
        self.save_benchmarks()
        
    def save_benchmarks(self) -> None:
        """Save all benchmarks to a JSON file."""
        with open(self.log_dir / "benchmarks.json", "w") as f:
            json.dump(self.benchmarks, f, indent=2)
            
    def load_benchmarks(self) -> None:
        """Load benchmarks from JSON file if it exists."""
        json_path = self.log_dir / "benchmarks.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                self.benchmarks = json.load(f)
                
    def compare_benchmarks(self, benchmarks: Optional[List[str]] = None) -> None:
        """
        Compare multiple benchmarks and print a comparison table.
        
        Args:
            benchmarks: List of benchmark names to compare. If None, compare all.
        """
        if not self.benchmarks:
            logging.info("No benchmarks to compare.")
            return
            
        if benchmarks:
            to_compare = [b for b in self.benchmarks if b["name"] in benchmarks]
        else:
            to_compare = self.benchmarks
            
        if not to_compare:
            logging.info("No matching benchmarks found.")
            return
            
        # Print header
        logging.info("\n" + "="*80)
        logging.info("BENCHMARK COMPARISON")
        logging.info("="*80)
        
        # Print table header
        header = f"{'Name':<30} | {'Throughput':<15} | {'Batch Size':<10} | {'Model Config':<20} | {'Optimizations':<20}"
        logging.info(header)
        logging.info("-"*len(header))
        
        # Print each benchmark
        for benchmark in to_compare:
            model_info = f"d={benchmark['model_config'].get('d_model', 'N/A')}, l={benchmark['model_config'].get('n_layers', 'N/A')}"
            opt_str = ", ".join([k[:3] for k, v in benchmark['optimizations'].items() if v])
            if not opt_str:
                opt_str = "None"
                
            row = f"{benchmark['name']:<30} | {benchmark['throughput']:<15.2f} | {benchmark['batch_size']:<10} | {model_info:<20} | {opt_str:<20}"
            logging.info(row)
            
        logging.info("="*80)
        
def get_benchmark_logger() -> BenchmarkLogger:
    """Get a singleton benchmark logger instance."""
    if not hasattr(get_benchmark_logger, "instance"):
        get_benchmark_logger.instance = BenchmarkLogger()
    return get_benchmark_logger.instance 