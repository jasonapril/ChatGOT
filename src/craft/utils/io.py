"""
I/O utility functions for file operations.

This module provides functions for reading and writing files,
particularly JSON configuration files.
"""
import json
import logging
import os
from typing import Any, Dict, Optional, cast, Union
import shutil
import time
from pathlib import Path
import requests
from tqdm import tqdm
import yaml

# Initialize logger at module level
logger = logging.getLogger(__name__)

def ensure_directory(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Cast the return value
            data = cast(Dict[str, Any], json.load(f))
        logging.info(f"Loaded JSON from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {str(e)}")
        raise


def save_json(data: Dict[str, Any], file_path: str, indent: int = 4) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary containing the data to save
        file_path: Path to save the JSON file
        indent: Number of spaces for indentation
    """
    # Create directory if it doesn't exist
    ensure_directory(os.path.dirname(file_path) if os.path.dirname(file_path) else ".")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        logging.info(f"Saved JSON to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {str(e)}")
        raise


def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size of the file in bytes
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logging.error(f"Failed to get size of {file_path}: {str(e)}")
        return 0


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.23 MB")
    """
    if size_bytes == 0:
        return "0B"
    
    # Define size units and their thresholds
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    
    # Use a float for calculations to avoid type conflict
    size_float = float(size_bytes)
    
    while size_float >= 1024 and unit_index < len(units) - 1:
        size_float /= 1024
        unit_index += 1
    
    return f"{size_float:.2f} {units[unit_index]}"


def create_output_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create an output directory based on a base directory and experiment name.
    
    Constructs the path as base_dir/experiment_name, ensures it exists,
    and returns the absolute path.
    
    Args:
        base_dir: The base directory for outputs.
        experiment_name: The name specific to this experiment/run.
        
    Returns:
        The absolute path to the created or ensured directory.
    """
    output_dir = os.path.join(base_dir, experiment_name)
    ensure_directory(output_dir) # Use the existing function to create if needed
    absolute_output_dir = os.path.abspath(output_dir)
    logging.info(f"Ensured output directory exists: {absolute_output_dir}")
    return absolute_output_dir


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Cast the result of yaml.safe_load
            # Assuming yaml is imported somewhere above
            # import yaml # Ensure yaml is imported # - Already imported above
            return cast(Dict[str, Any], yaml.safe_load(f))
    except FileNotFoundError as e: # Modified this block
        logger.error(f"YAML file not found: {file_path}")
        raise e # Re-raise the exception
    except Exception as e:
        logger.error(f"Failed to load YAML file {file_path}: {e}")
        raise e # Re-raise other exceptions too


def download_file(url: str, filename: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length is None:
        # Cannot determine total size
        # logger = logging.getLogger(__name__) # Removed: Use module-level logger
        logger.warning("Content length not found in headers, cannot show progress.")
        pbar = tqdm(unit="B", unit_scale=True, desc=filename)
        total_size = -1 # Indicate unknown size
    else:
        # Cast content_length to int after converting to float
        total_size = int(float(content_length))
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=filename)

    chunk_size = 8192 # Read 8KB at a time

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    pbar.close()

    if total_size == -1:
        # logger = logging.getLogger(__name__) # Removed: Use module-level logger
        logger.info(f"Downloaded {filename} (unknown size)")
    else:
        # logger = logging.getLogger(__name__) # Removed: Use module-level logger
        logger.info(f"Downloaded {filename} ({format_file_size(total_size)})") 