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
import sys # Import sys for stderr printing

OUTPUTS_DIR = "outputs" # Base directory for Hydra runs
METADATA_FILE = "experiment_info.json"
CHECKPOINT_DIR_NAME = "checkpoints"
CHECKPOINT_PATTERN = "checkpoint_step_*.pt"

def parse_args():
    parser = argparse.ArgumentParser(description="Find the latest experiment checkpoint.")
    parser.add_argument("--model-architecture", type=str, help="Filter runs by model architecture.")
    parser.add_argument("--dataset-target", type=str, help="Filter runs by dataset target class.")
    parser.add_argument("--experiment-name", type=str, help="Filter runs by experiment name.")
    parser.add_argument("--base-dir", type=str, default="outputs/hydra", help="Base directory containing Hydra runs.")
    # Add more filter arguments as needed (e.g., --experiment-name)
    return parser.parse_args()

def find_matching_run_dirs_sorted(filter_args):
    """Finds all run directories matching filter criteria, sorted newest first."""
    matching_runs = [] # Store tuples of (datetime, path)
    base_search_dir = filter_args.base_dir

    if not os.path.isdir(base_search_dir):
        return []

    date_dirs = sorted(glob.glob(os.path.join(base_search_dir, "*-*-*")), reverse=True)

    for date_dir in date_dirs:
        if not os.path.isdir(date_dir):
            continue
        
        time_dirs = sorted(glob.glob(os.path.join(date_dir, "*-*-*")), reverse=True)
        
        for run_dir in time_dirs:
            if not os.path.isdir(run_dir):
                continue
            
            try:
                run_time_str = os.path.basename(run_dir)
                date_str = os.path.basename(date_dir)
                run_datetime = datetime.strptime(f"{date_str} {run_time_str}", "%Y-%m-%d %H-%M-%S")
            except ValueError:
                continue 

            metadata_path = os.path.join(run_dir, METADATA_FILE)
            if not os.path.exists(metadata_path):
                continue 

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                continue 

            # Apply filters
            match = True
            if filter_args.experiment_name and metadata.get("experiment_name") != filter_args.experiment_name:
                match = False
            if filter_args.model_architecture and metadata.get("model_architecture") != filter_args.model_architecture:
                match = False
            if filter_args.dataset_target and metadata.get("dataset_target") != filter_args.dataset_target:
                match = False
            
            if match:
                matching_runs.append((run_datetime, run_dir))
                
    # Sort by datetime descending (newest first)
    matching_runs.sort(key=lambda item: item[0], reverse=True)
    
    # Return only the paths, sorted newest first
    sorted_run_dirs = [run_path for dt, run_path in matching_runs]
    return sorted_run_dirs

def find_latest_checkpoint_in_dir(run_dir):
    """Finds the checkpoint with the highest step number in a specific run directory."""
    checkpoint_dir = os.path.join(run_dir, CHECKPOINT_DIR_NAME)
    if not os.path.isdir(checkpoint_dir):
        return None, -1 # Return path and step number

    checkpoints = glob.glob(os.path.join(checkpoint_dir, CHECKPOINT_PATTERN))
    latest_checkpoint_path = None
    max_step = -1

    step_regex = re.compile(r"checkpoint_step_(\d+)\.pt")

    for ckpt_path in checkpoints:
        match = step_regex.search(os.path.basename(ckpt_path))
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                latest_checkpoint_path = ckpt_path

    return latest_checkpoint_path, max_step

if __name__ == "__main__":
    args = parse_args()
    
    # Find all matching runs, sorted newest first
    matching_run_dirs = find_matching_run_dirs_sorted(args)

    overall_best_checkpoint_path = None
    overall_max_step = -1

    if matching_run_dirs:
        # Iterate through matching runs to find the overall best checkpoint
        for run_dir in matching_run_dirs:
            latest_checkpoint_path, max_step = find_latest_checkpoint_in_dir(run_dir)
            
            if latest_checkpoint_path and max_step > overall_max_step:
                overall_max_step = max_step
                overall_best_checkpoint_path = latest_checkpoint_path

        # After checking all matching runs, print the overall best
        if overall_best_checkpoint_path:
            abs_path = os.path.abspath(overall_best_checkpoint_path)
            print(abs_path) # Print path to stdout for the calling script
        else:
            pass # Exit silently if no checkpoint found
    else:
        pass # Exit silently if no run found 