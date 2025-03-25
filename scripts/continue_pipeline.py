#!/usr/bin/env python
"""
ChatGoT Pipeline Continuation Script

This script provides a convenient way to continue the pipeline
from a specific stage, with validation checks along the way.

Usage:
    python continue_pipeline.py --stage <stage_name>
    
Available stages:
    process    - Process raw data into training format
    optimize   - Optimize hyperparameters
    train      - Train the model
    generate   - Generate text from the model
    all        - Run all pipeline stages
"""

import os
import sys
import argparse
import logging
import time
import subprocess
from pathlib import Path

# Get the python executable used to run this script
python_exe = sys.executable

def execute_command(cmd, stage_name):
    """Execute a command and handle errors."""
    print(f"\n{'-'*80}")
    print(f"Starting {stage_name} stage...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'-'*80}\n")
    
    start_time = time.time()
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream the output
        for line in process.stdout:
            print(line, end='')
        
        # Wait for the process to complete
        returncode = process.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if returncode != 0:
            print(f"\n{'-'*80}")
            print(f"Error: {stage_name} stage failed with code {returncode}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"{'-'*80}\n")
            sys.exit(returncode)
        
        print(f"\n{'-'*80}")
        print(f"{stage_name} stage completed successfully")
        print(f"Duration: {duration:.2f} seconds")
        print(f"{'-'*80}\n")
        
    except Exception as e:
        print(f"Error executing {stage_name} stage: {e}")
        sys.exit(1)

def run_process_stage():
    """Run the data processing stage."""
    # For backwards compatibility, use the old command structure
    cmd = [python_exe, "-m", "src.cli.main", "process", "--config", "default"]
    execute_command(cmd, "Process")

def run_train_stage(skip_tests=False):
    """Run the training stage."""
    # For backwards compatibility, use the old command structure
    cmd = [
        python_exe, 
        "-m", 
        "src.cli.main",
        "train", 
        "--config", 
        "default"
    ]
    if skip_tests:
        cmd.append("--skip-tests")
    execute_command(cmd, "Train")

def run_generate_stage():
    """Run the generation stage."""
    cmd = [python_exe, "-m", "src.cli.main", "generate", "--config", "default"]
    execute_command(cmd, "Generate")

def run_all_stages(skip_tests=False):
    """Run all pipeline stages in sequence."""
    run_process_stage()
    run_train_stage(skip_tests)
    run_generate_stage()

def run_tests():
    """Run tests to ensure the project is in a good state."""
    logging.info("Running tests before continuing pipeline...")
    
    try:
        # Change to the tests directory
        os.chdir('tests')
        
        # Run unit tests only - these should be fast
        result = subprocess.run(
            ["python", "run_all_tests.py", "--unit-only"],
            capture_output=True,
            text=True
        )
        
        # Change back to the original directory
        os.chdir('..')
        
        if result.returncode != 0:
            logging.error("Unit tests failed. Pipeline cannot continue.")
            logging.error(result.stdout)
            logging.error(result.stderr)
            return False
        
        logging.info("Tests passed successfully.")
        return True
        
    except Exception as e:
        logging.error(f"Error running tests: {e}")
        # Change back to the original directory in case of error
        try:
            os.chdir('..')
        except:
            pass
        return False

def verify_data_processed():
    """Verify that data has been processed."""
    processed_file = Path("processed_data/got_char_data.pkl")
    vocab_file = Path("processed_data/vocab.json")
    
    if processed_file.exists() and vocab_file.exists():
        logging.info("Processed data found.")
        return True
    
    logging.error("Processed data not found. Run the 'process' stage first.")
    return False

def run_stage(stage_name):
    """Run a specific pipeline stage."""
    logging.info(f"Running pipeline stage: {stage_name.upper()}")
    
    # Print Python environment info
    logging.info("Python executable path: %s", python_exe)
    logging.info("Python path: %s", sys.path)
    
    # Run the appropriate command for each stage
    if stage_name == "process":
        cmd = [python_exe, "-m", "src.cli.main", "process", "--config", "default"]
    elif stage_name == "train":
        # Use basic command with correct Hydra override syntax
        cmd = [
            python_exe, 
            "-m", 
            "src.cli.main", 
            "train", 
            "--config", 
            "default",
            "+pipeline.stages=[train]"  # Using + to append to non-existent key
        ]
    elif stage_name == "generate":
        cmd = [python_exe, "-m", "src.cli.main", "generate", "--config", "default"]
    else:
        logging.error(f"Unknown stage: {stage_name}")
        return False
    
    # Run the command
    logging.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logging.error(f"Stage '{stage_name}' failed with return code {result.returncode}")
        return False
    
    logging.info(f"Stage '{stage_name}' completed successfully.")
    return True

def run_validation():
    """Run model validation."""
    logging.info("Running model validation...")
    
    cmd = ["python", "scripts/validate_model.py", "--latest", "--save"]
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        logging.error("Model validation failed.")
        return False
    
    logging.info("Model validation completed successfully.")
    return True

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Continue ChatGoT pipeline from a specific stage.")
    parser.add_argument("--stage", choices=["process", "train", "generate", "all"],
                        required=True, help="Pipeline stage to continue from")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-validation", action="store_true", help="Skip model validation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Starting pipeline continuation script")
    logging.info(f"Stage: {args.stage}")
    logging.info(f"Skip tests: {args.skip_tests}")
    logging.info(f"Skip validation: {args.skip_validation}")
    
    # Run tests unless skipped
    if not args.skip_tests:
        if not run_tests():
            return 1
    
    # Define the stages in order
    stages = ["process", "train", "generate"]
    
    # Determine which stages to run
    if args.stage == "all":
        start_index = 0
    else:
        try:
            start_index = stages.index(args.stage)
        except ValueError:
            print(f"Unknown stage: {args.stage}")
            return 1
    
    # Run the stages in order
    for i in range(start_index, len(stages)):
        stage = stages[i]
        
        # Special checks for certain stages
        if stage == "train":
            if not verify_data_processed():
                return 1
        
        # Run the stage
        if not run_stage(stage):
            return 1
        
        # Run validation after training unless skipped
        if stage == "train" and not args.skip_validation:
            if not run_validation():
                # Validation failing is not fatal to the pipeline
                logging.warning("Model validation failed but continuing pipeline.")
    
    logging.info("Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 