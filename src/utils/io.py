"""
I/O utility functions for file operations.

This module provides functions for reading and writing files,
particularly JSON configuration files.
"""
import json
import logging
import os
from typing import Any, Dict


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
            data = json.load(f)
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
    
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024
        unit_index += 1
    
    return f"{size_bytes:.2f} {units[unit_index]}" 