#!/usr/bin/env python
"""
Benchmark Runner
===============

This module provides functionality for running benchmarks against the
text generation pipeline. It supports running multiple benchmarks
with different configurations and collecting results.

Features:
- Configurable benchmark suite execution
- Performance measurement
- Results aggregation and reporting
- Comparison against baselines
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Tuple
import importlib
from datetime import datetime

# Add the parent directory to the path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import setup_logger, log_section_header
from src.utils.device import get_device_info

class BenchmarkRunner:
    """Runner for executing benchmark suites."""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the benchmark runner.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.benchmark_dir = args.output_dir
        os.makedirs(self.benchmark_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(self.benchmark_dir, "benchmark.log")
        setup_logger(log_file, args.log_level)
        
        self.results = {}
        self.baseline_results = self._load_baseline()
        
    def _load_baseline(self) -> Dict[str, Any]:
        """
        Load baseline results from file if available.
        
        Returns:
            Dictionary of baseline results or empty dict if not available
        """
        if not self.args.baseline_file:
            return {}
            
        if not os.path.exists(self.args.baseline_file):
            logging.warning(f"Baseline file {self.args.baseline_file} not found")
            return {}
            
        try:
            with open(self.args.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load baseline file: {e}")
            return {}
    
    def _discover_benchmarks(self) -> List[str]:
        """
        Discover available benchmarks in the benchmarks directory.
        
        Returns:
            List of benchmark module names
        """
        benchmark_path = os.path.join(os.path.dirname(__file__), "benchmarks")
        benchmarks = []
        
        # Check if specific benchmarks were requested
        if self.args.benchmarks:
            for benchmark in self.args.benchmarks:
                benchmark_file = os.path.join(benchmark_path, f"{benchmark}.py")
                if os.path.exists(benchmark_file):
                    benchmarks.append(benchmark)
                else:
                    logging.warning(f"Benchmark {benchmark} not found")
            return benchmarks
            
        # Discover all benchmarks
        for file in os.listdir(benchmark_path):
            if file.endswith(".py") and not file.startswith("_"):
                benchmark = file[:-3]  # Remove .py extension
                benchmarks.append(benchmark)
                
        return sorted(benchmarks)
        
    def run(self) -> Dict[str, Any]:
        """
        Run all benchmarks and collect results.
        
        Returns:
            Dictionary of benchmark results
        """
        logging.info("Starting benchmark runner")
        
        # Record system information
        device_info = get_device_info()
        self.results["system_info"] = {
            "device": device_info["device_name"],
            "cuda_version": device_info.get("cuda_version", "N/A"),
            "cuda_devices": device_info.get("cuda_devices", []),
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
        }
        
        # Discover benchmarks
        benchmarks = self._discover_benchmarks()
        if not benchmarks:
            logging.warning("No benchmarks found!")
            return self.results
            
        logging.info(f"Discovered benchmarks: {', '.join(benchmarks)}")
        
        # Import and run benchmarks
        benchmark_results = {}
        for benchmark_name in benchmarks:
            log_section_header(f"BENCHMARK: {benchmark_name}")
            
            try:
                # Import the benchmark module
                benchmark_module = importlib.import_module(f"benchmarking.benchmarks.{benchmark_name}")
                
                # Run benchmark
                logging.info(f"Running benchmark: {benchmark_name}")
                start_time = time.time()
                result = benchmark_module.run_benchmark(self.args)
                elapsed = time.time() - start_time
                
                # Add execution time
                result["execution_time"] = elapsed
                
                # Compare with baseline if available
                if benchmark_name in self.baseline_results:
                    baseline = self.baseline_results[benchmark_name]
                    result["comparison"] = self._compare_with_baseline(result, baseline)
                
                benchmark_results[benchmark_name] = result
                logging.info(f"Completed benchmark {benchmark_name} in {elapsed:.2f} seconds")
                
            except Exception as e:
                logging.error(f"Failed to run benchmark {benchmark_name}: {e}", exc_info=True)
                benchmark_results[benchmark_name] = {"status": "failed", "error": str(e)}
        
        self.results["benchmarks"] = benchmark_results
        self._save_results()
        
        return self.results
    
    def _compare_with_baseline(self, result: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare benchmark results with baseline.
        
        Args:
            result: Current benchmark results
            baseline: Baseline results for comparison
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison = {}
        
        # Compare metrics if they exist in both result and baseline
        for key in result.keys():
            if key in baseline and isinstance(result[key], (int, float)) and key != "execution_time":
                baseline_value = baseline[key]
                current_value = result[key]
                
                # Calculate difference and percent change
                diff = current_value - baseline_value
                if baseline_value != 0:
                    percent = (diff / baseline_value) * 100
                    comparison[f"{key}_change_pct"] = percent
                
                comparison[f"{key}_change"] = diff
                comparison[f"{key}_baseline"] = baseline_value
        
        # Compare execution time
        if "execution_time" in result and "execution_time" in baseline:
            exec_diff = result["execution_time"] - baseline["execution_time"]
            exec_pct = (exec_diff / baseline["execution_time"]) * 100 if baseline["execution_time"] != 0 else 0
            comparison["execution_time_change"] = exec_diff
            comparison["execution_time_change_pct"] = exec_pct
            comparison["execution_time_baseline"] = baseline["execution_time"]
        
        return comparison
    
    def _save_results(self) -> None:
        """Save benchmark results to file."""
        results_file = os.path.join(self.benchmark_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"Results saved to {results_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ChatGoT Benchmark Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, default='./benchmark_results/{date}',
                        help='Directory to store benchmark results')
    parser.add_argument('--benchmarks', type=str, nargs='+',
                        help='Specific benchmarks to run (default: all)')
    parser.add_argument('--baseline-file', type=str,
                        help='Baseline results file for comparison')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing benchmark data')
    parser.add_argument('--model-checkpoint', type=str,
                        help='Model checkpoint file to use for inference benchmarks')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations to run each benchmark')
    
    args = parser.parse_args()
    
    # Process the date template in the output directory
    if '{date}' in args.output_dir:
        date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = args.output_dir.replace('{date}', date_str)
    
    return args

def main():
    """Main entry point for the benchmark runner."""
    args = parse_args()
    
    runner = BenchmarkRunner(args)
    results = runner.run()
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-----------------")
    for name, result in results.get("benchmarks", {}).items():
        status = "✅ Passed" if result.get("status") != "failed" else "❌ Failed"
        time = f"{result.get('execution_time', 0):.2f}s"
        print(f"{name: <20} {status: <10} {time: <10}")
        
        # Print comparison if available
        if "comparison" in result:
            comp = result["comparison"]
            for key, value in comp.items():
                if key.endswith("_change_pct") and not key.startswith("execution_time"):
                    metric = key.replace("_change_pct", "")
                    direction = "faster" if value < 0 else "slower" if value > 0 else "same"
                    print(f"  {metric: <15} {abs(value):.2f}% {direction}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 