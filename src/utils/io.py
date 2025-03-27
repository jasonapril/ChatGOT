"""
I/O utilities for Craft.

This module provides utilities for input/output operations, such as file handling,
directory creation, and saving/loading of configuration files.
"""
import os
import json
import logging
import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path


def create_output_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create an output directory with an optional experiment name and timestamp.
    
    Args:
        base_dir: Base directory for outputs
        experiment_name: Optional experiment name to include in the path
        
    Returns:
        Path to the created output directory
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a timestamped directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{experiment_name}_{timestamp}"
    else:
        dir_name = timestamp
    
    # Create the full output path
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Created output directory: {output_dir}")
    return output_dir


def save_args(args: Any, output_path: str) -> None:
    """
    Save command-line arguments or configuration to a JSON file.
    
    Args:
        args: Arguments to save (can be argparse.Namespace or dict)
        output_path: Path to save the arguments
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert args to dictionary if it's not already
    if not isinstance(args, dict):
        # Handle argparse.Namespace
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = {k: v for k, v in args.__dict__.items() 
                        if not k.startswith('_')}
    else:
        args_dict = args
    
    # Convert any non-serializable objects to strings
    serializable_dict = {}
    for k, v in args_dict.items():
        if isinstance(v, (str, int, float, bool, list, dict, tuple, type(None))):
            serializable_dict[k] = v
        else:
            serializable_dict[k] = str(v)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    
    logging.info(f"Arguments saved to {output_path}")


def load_json(file_path: str) -> Dict:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON as a dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data: Dict, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: Indentation level for the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size of the file in bytes
    """
    return os.path.getsize(file_path)


def format_file_size(size_in_bytes: int) -> str:
    """
    Format file size in a human-readable format.
    
    Args:
        size_in_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024 or unit == 'TB':
            break
        size_in_bytes /= 1024.0
    
    return f"{size_in_bytes:.2f} {unit}" 