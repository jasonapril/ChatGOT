#!/usr/bin/env python
"""
Test Suite for Optimization Features
===================================

This script runs all tests and benchmarks related to optimization features,
specifically focusing on the PyTorch 2.0+ Compilation optimization.
"""

import unittest
import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all optimization tests and benchmarks."
    )
    
    parser.add_argument("--skip_unit_tests", action="store_true",
                        help="Skip unit tests")
    parser.add_argument("--skip_integration_tests", action="store_true",
                        help="Skip integration tests")
    parser.add_argument("--skip_benchmarks", action="store_true",
                        help="Skip benchmarks")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to processed data for benchmarks (required if running benchmarks)")
    parser.add_argument("--benchmark_epochs", type=int, default=1,
                        help="Number of epochs for benchmark tests")
    parser.add_argument("--benchmark_iterations", type=int, default=2,
                        help="Number of iterations for benchmark tests")
    
    return parser.parse_args()

def run_unit_tests():
    """Run all unit tests for optimization features."""
    logging.info("Running unit tests for optimization features...")
    
    test_loader = unittest.TestLoader()
    
    # Load tests from test_optimizations.py
    test_path = os.path.join(os.path.dirname(__file__), 'unit', 'test_optimizations.py')
    test_suite = test_loader.discover(os.path.dirname(test_path), pattern=os.path.basename(test_path))
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run integration tests for optimization features."""
    logging.info("Running integration tests for optimization features...")
    
    test_loader = unittest.TestLoader()
    
    # Load tests from test_optimizations.py in integration directory
    test_path = os.path.join(os.path.dirname(__file__), 'integration', 'test_optimizations.py')
    test_suite = test_loader.discover(os.path.dirname(test_path), pattern=os.path.basename(test_path))
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

def run_benchmarks(data_path, epochs=1, iterations=2):
    """Run benchmarks for optimization features."""
    if not data_path:
        logging.error("Data path required for benchmarks!")
        return False
    
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return False
    
    logging.info("Running benchmarks for optimization features...")
    
    # Path to benchmark script
    benchmark_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'benchmarks', 'benchmark_torch_compile.py')
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run benchmark command
    cmd = [
        sys.executable,
        benchmark_script,
        "--data_path", data_path,
        "--epochs", str(epochs),
        "--iterations", str(iterations),
        "--output_dir", output_dir
    ]
    
    logging.info(f"Running benchmark command: {' '.join(cmd)}")
    
    try:
        # Run the benchmark process
        process = subprocess.run(cmd, check=True)
        
        if process.returncode == 0:
            logging.info("Benchmarks completed successfully!")
            logging.info(f"Results saved to: {output_dir}")
            return True
        else:
            logging.error(f"Benchmark process failed with return code: {process.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Benchmark process failed: {e}")
        return False
    except Exception as e:
        logging.error(f"Error running benchmarks: {e}")
        return False

def main():
    """Main function to run all tests and benchmarks."""
    args = parse_args()
    
    success = True
    
    # Run all tests and benchmarks
    if not args.skip_unit_tests:
        if not run_unit_tests():
            logging.warning("Unit tests failed!")
            success = False
    else:
        logging.info("Skipping unit tests...")
    
    if not args.skip_integration_tests:
        if not run_integration_tests():
            logging.warning("Integration tests failed!")
            success = False
    else:
        logging.info("Skipping integration tests...")
    
    if not args.skip_benchmarks:
        if not args.data_path:
            logging.error("Data path is required for benchmarks! Use --data_path option.")
            success = False
        else:
            if not run_benchmarks(args.data_path, args.benchmark_epochs, args.benchmark_iterations):
                logging.warning("Benchmarks failed!")
                success = False
    else:
        logging.info("Skipping benchmarks...")
    
    # Provide final summary
    if success:
        logging.info("All tests and benchmarks completed successfully!")
        return 0
    else:
        logging.error("Some tests or benchmarks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 