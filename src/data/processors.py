"""
Data processing utilities for Craft.

This module provides functions for processing and preparing different types of datasets.
"""
import os
import json
import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from tqdm import tqdm


def prepare_data(
    input_file: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prepare a dataset for training.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory to save processed data
        config: Configuration dictionary
        
    Returns:
        Path to the processed dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data type from config or infer from file extension
    data_type = None
    if config:
        data_type = config.get("data", {}).get("type", None)
    
    if data_type is None:
        # Infer data type from file extension
        if input_file.endswith((".txt", ".text")):
            data_type = "text"
        elif input_file.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            data_type = "image"
        elif input_file.endswith((".wav", ".mp3")):
            data_type = "audio"
        elif input_file.endswith((".json", ".jsonl")):
            data_type = "json"
        else:
            raise ValueError(f"Cannot infer data type from file extension: {input_file}")
    
    # Process based on data type
    if data_type == "text":
        return prepare_text_data(input_file, output_dir, config)
    elif data_type == "image":
        return prepare_image_data(input_file, output_dir, config)
    elif data_type == "audio":
        return prepare_audio_data(input_file, output_dir, config)
    elif data_type == "json":
        return prepare_json_data(input_file, output_dir, config)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def prepare_text_data(
    input_file: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prepare text data for training.
    
    Args:
        input_file: Path to the input text file
        output_dir: Directory to save processed data
        config: Configuration dictionary
        
    Returns:
        Path to the processed dataset
    """
    logging.info(f"Processing text data from {input_file}")
    
    # Get configuration
    if config is None:
        config = {}
    
    text_format = config.get("data", {}).get("format", "character")
    
    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    logging.info(f"Read {len(text)} characters from {input_file}")
    
    # Process based on format
    if text_format == "character":
        # Character-level processing
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        data = {
            "text": text,
            "chars": chars,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "vocab_size": len(chars)
        }
        
        logging.info(f"Vocabulary size: {len(chars)} unique characters")
    else:
        raise ValueError(f"Unsupported text format: {text_format}")
    
    # Create output filename
    output_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{output_filename}_processed.pkl")
    
    # Save processed data
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    logging.info(f"Processed data saved to {output_path}")
    
    return output_path


def prepare_image_data(
    input_file: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prepare image data for training.
    
    This is a placeholder for future implementation.
    
    Args:
        input_file: Path to the input image file or directory
        output_dir: Directory to save processed data
        config: Configuration dictionary
        
    Returns:
        Path to the processed dataset
    """
    raise NotImplementedError("Image data processing not yet implemented")


def prepare_audio_data(
    input_file: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prepare audio data for training.
    
    This is a placeholder for future implementation.
    
    Args:
        input_file: Path to the input audio file or directory
        output_dir: Directory to save processed data
        config: Configuration dictionary
        
    Returns:
        Path to the processed dataset
    """
    raise NotImplementedError("Audio data processing not yet implemented")


def prepare_json_data(
    input_file: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prepare JSON data for training.
    
    This can process various formats like json/jsonl files.
    
    Args:
        input_file: Path to the input JSON file
        output_dir: Directory to save processed data
        config: Configuration dictionary
        
    Returns:
        Path to the processed dataset
    """
    logging.info(f"Processing JSON data from {input_file}")
    
    # Get configuration
    if config is None:
        config = {}
    
    # Determine if it's JSONL format
    is_jsonl = input_file.endswith(".jsonl") or config.get("data", {}).get("format") == "jsonl"
    
    # Read input file
    if is_jsonl:
        # Read line by line for JSONL
        items = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        # Read entire file for JSON
        with open(input_file, "r", encoding="utf-8") as f:
            items = json.load(f)
    
    logging.info(f"Read {len(items)} items from {input_file}")
    
    # Create output filename
    output_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{output_filename}_processed.pkl")
    
    # Save processed data
    with open(output_path, "wb") as f:
        pickle.dump(items, f)
    
    logging.info(f"Processed data saved to {output_path}")
    
    return output_path


def split_data(
    data: List[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: List of data items
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Check ratios
    assert 0 <= train_ratio <= 1, "train_ratio must be between 0 and 1"
    assert 0 <= val_ratio <= 1, "val_ratio must be between 0 and 1"
    assert 0 <= test_ratio <= 1, "test_ratio must be between 0 and 1"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    # Set random seed
    np.random.seed(seed)
    
    # Shuffle data
    indices = np.random.permutation(len(data))
    
    # Calculate split indices
    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)
    
    # Split data
    train_data = [data[i] for i in indices[:train_end]]
    val_data = [data[i] for i in indices[train_end:val_end]]
    test_data = [data[i] for i in indices[val_end:]]
    
    return train_data, val_data, test_data 