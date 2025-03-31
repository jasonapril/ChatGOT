#!/usr/bin/env python
"""
Benchmarking CLI Entry Point
===========================

This is the main command-line interface for benchmarking tools, providing
access to both benchmark running and visualization tools.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.runner import BenchmarkRunner, parse_args as runner_parse_args
from benchmarking.utils.visualization import create_benchmark_report

def parse_args():
    """Parse command line arguments for the main benchmarking CLI."""
    parser = argparse.ArgumentParser(
        description='Craft Benchmarking Tools',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Benchmarking command')
    
    # Run benchmarks command
    run_parser = subparsers.add_parser('run', help='Run benchmarks')
    run_parser.add_argument('--output-dir', type=str, default='./outputs/benchmarks/{date}',
                        help='Directory to store benchmark results')
    run_parser.add_argument('--benchmarks', type=str, nargs='+',
                        help='Specific benchmarks to run (default: all)')
    run_parser.add_argument('--baseline-file', type=str,
                        help='Baseline results file for comparison')
    run_parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    run_parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing benchmark data')
    run_parser.add_argument('--model-checkpoint', type=str,
                        help='Model checkpoint file to use for inference benchmarks')
    run_parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations to run each benchmark')
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate benchmark report')
    report_parser.add_argument('--results-file', type=str, required=True,
                          help='Path to benchmark results JSON file')
    report_parser.add_argument('--output-dir', type=str, default='./outputs/benchmarks/reports/{date}',
                          help='Directory to save the report files')
    report_parser.add_argument('--comparison-file', type=str,
                          help='Path to baseline results for comparison')
    
    # List available benchmarks command
    list_parser = subparsers.add_parser('list', help='List available benchmarks')
    
    args = parser.parse_args()
    
    # Process date template in directories
    if args.command == 'run' or args.command == 'report':
        if '{date}' in args.output_dir:
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output_dir = args.output_dir.replace('{date}', date_str)
    
    return args

def list_available_benchmarks():
    """List all available benchmarks."""
    benchmark_path = os.path.join(os.path.dirname(__file__), "benchmarks")
    if not os.path.exists(benchmark_path):
        print("Benchmark directory not found!")
        return
    
    benchmarks = []
    for file in os.listdir(benchmark_path):
        if file.endswith(".py") and not file.startswith("_"):
            benchmark = file[:-3]  # Remove .py extension
            benchmarks.append(benchmark)
    
    if not benchmarks:
        print("No benchmarks found!")
        return
    
    print("\nAvailable Benchmarks:")
    print("=====================")
    for benchmark in sorted(benchmarks):
        print(f"- {benchmark}")
    
    print("\nTo run a specific benchmark:")
    print("python -m benchmarking.main run --benchmarks benchmark_name")
    print("\nTo run all benchmarks:")
    print("python -m benchmarking.main run")

def main():
    """Main entry point for the benchmarking tools."""
    args = parse_args()
    
    if args.command == 'run':
        # Run benchmarks using the benchmark runner
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
        
    elif args.command == 'report':
        # Generate report from existing results
        report_file = create_benchmark_report(
            args.results_file,
            args.output_dir,
            args.comparison_file
        )
        
        print(f"\nReport generated at: {report_file}")
        return 0
        
    elif args.command == 'list':
        # List available benchmarks
        list_available_benchmarks()
        return 0
        
    else:
        # No command provided, show help
        parse_args(['--help'])
        return 1

if __name__ == '__main__':
    sys.exit(main()) 