"""
Script to find the latest checkpoint path for a specific experiment.

Scans Hydra output directories, reads experiment_info.json,
filters based on provided criteria (e.g., model, dataset),
and prints the path to the checkpoint with the highest step number
in the latest matching run directory.
"""

import os
import json
import argparse
import glob
import re
from datetime import datetime

OUTPUTS_DIR = "outputs" # Base directory for Hydra runs
METADATA_FILE = "experiment_info.json"
CHECKPOINT_DIR_NAME = "checkpoints"
CHECKPOINT_PATTERN = "checkpoint_step_*.pt"

def parse_args():
    parser = argparse.ArgumentParser(description="Find the latest experiment checkpoint.")
    parser.add_argument("--model-architecture", type=str, help="Filter runs by model architecture.")
    parser.add_argument("--dataset-target", type=str, help="Filter runs by dataset target class.")
    parser.add_argument("--experiment-name", type=str, help="Filter runs by experiment name.")
    # Add more filter arguments as needed (e.g., --experiment-name)
    return parser.parse_args()

def find_latest_run_dir(filter_args):
    latest_run_dir = None
    latest_run_time = datetime.min

    if not os.path.isdir(OUTPUTS_DIR):
        # print(f"Outputs directory not found: {OUTPUTS_DIR}", file=sys.stderr)
        return None

    date_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "*-*-*")), reverse=True)

    for date_dir in date_dirs:
        if not os.path.isdir(date_dir):
            continue
        
        time_dirs = sorted(glob.glob(os.path.join(date_dir, "*-*-*")), reverse=True)
        
        for run_dir in time_dirs:
            if not os.path.isdir(run_dir):
                continue
            
            try:
                run_time = datetime.strptime(os.path.basename(run_dir), "%H-%M-%S")
                # Combine date and time - simplistic, assumes dir structure YYYY-MM-DD/HH-MM-SS
                run_datetime = datetime.combine(datetime.strptime(os.path.basename(date_dir), "%Y-%m-%d").date(), run_time.time())
            except ValueError:
                continue # Skip directories with unexpected names

            metadata_path = os.path.join(run_dir, METADATA_FILE)
            if not os.path.exists(metadata_path):
                continue # Skip runs without metadata

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                continue # Skip runs with invalid metadata

            # Apply filters
            match = True
            if filter_args.experiment_name and metadata.get("experiment_name") != filter_args.experiment_name:
                match = False
            if filter_args.model_architecture and metadata.get("model_architecture") != filter_args.model_architecture:
                match = False
            if filter_args.dataset_target and metadata.get("dataset_target") != filter_args.dataset_target:
                match = False
            # Add more filter checks here
            
            if match:
                # Found the latest matching run so far
                return run_dir 
                # # If we wanted the absolute latest, we'd compare run_datetime
                # if run_datetime > latest_run_time:
                #     latest_run_time = run_datetime
                #     latest_run_dir = run_dir
                
    return None # No matching run found

def find_latest_checkpoint(run_dir):
    checkpoint_dir = os.path.join(run_dir, CHECKPOINT_DIR_NAME)
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, CHECKPOINT_PATTERN))
    latest_checkpoint = None
    max_step = -1

    step_regex = re.compile(r"checkpoint_step_(\d+)\.pt")

    for ckpt in checkpoints:
        match = step_regex.search(os.path.basename(ckpt))
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                latest_checkpoint = ckpt

    return latest_checkpoint

if __name__ == "__main__":
    args = parse_args()
    latest_run_dir = find_latest_run_dir(args)

    if latest_run_dir:
        latest_checkpoint_path = find_latest_checkpoint(latest_run_dir)
        if latest_checkpoint_path:
            # Print the relative path from the assumed execution directory (project root)
            print(os.path.relpath(latest_checkpoint_path))
        else:
            # print(f"No checkpoints found in latest matching run: {latest_run_dir}", file=sys.stderr)
            pass # Exit silently if no checkpoint found
    else:
        # print("No matching run directory found.", file=sys.stderr)
        pass # Exit silently if no run found 