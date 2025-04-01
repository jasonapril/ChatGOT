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

# Import AutoTokenizer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

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
    
    # Process data based on type
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
    config: Dict = None
) -> Dict[str, str]:
    """
    Prepare text data for training by tokenizing and splitting into train/val/test sets.

    Args:
        input_file (str): Path to input text file.
        output_dir (str): Directory to save processed data.
        config (Dict): Configuration dictionary containing data processing parameters.

    Returns:
        Dict[str, str]: Dictionary containing paths to train/val/test splits.
    """
    if not config:
        config = {}

    data_config = config.get('data', {})
    text_format = data_config.get('format', 'character')
    if text_format not in ['character', 'subword', 'pretrained']:
        raise ValueError(f"Unsupported text format: {text_format}")

    # Load text data
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process based on format
    if text_format == 'character':
        # Character-level processing
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        token_ids = [char_to_idx[ch] for ch in text]
        data_dict = {
            'text': text,
            'chars': chars,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'vocab_size': vocab_size,
            'token_ids': np.array(token_ids, dtype=np.uint16),
            'tokenizer_type': 'character'
        }
    else:
        # Subword or pretrained tokenizer
        tokenizer_name = data_config.get('tokenizer_name')
        if not tokenizer_name:
            raise ValueError("tokenizer_name must be provided for subword or pretrained tokenization")

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            token_ids = tokenizer.encode(text)
            data_dict = {
                'text': text,
                'token_ids': np.array(token_ids, dtype=np.uint16),
                'vocab_size': tokenizer.vocab_size,
                'tokenizer_name': tokenizer_name,
                'tokenizer_type': text_format
            }
        except Exception as e:
            logger.error(f"Failed to load or use tokenizer {tokenizer_name}: {e}")
            raise

    # Validate split ratios
    split_ratios = data_config.get('split_ratios', [0.7, 0.15, 0.15])
    if not isinstance(split_ratios, list) or len(split_ratios) != 3 or sum(split_ratios) != 1.0:
        raise ValueError("split_ratios must be a list of 3 numbers that sum to 1.0")

    # Calculate split sizes
    total_size = len(data_dict['token_ids'])
    train_size = int(total_size * split_ratios[0])
    val_size = int(total_size * split_ratios[1])
    test_size = total_size - train_size - val_size

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    output_paths = {}
    splits = {
        'train': (0, train_size),
        'val': (train_size, train_size + val_size),
        'test': (train_size + val_size, total_size)
    }

    for split_name, (start, end) in splits.items():
        split_data = data_dict.copy()
        split_data['token_ids'] = data_dict['token_ids'][start:end]
        output_path = os.path.join(output_dir, f"{split_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(split_data, f)
        output_paths[split_name] = output_path

    logger.info(f"Saved splits to {output_dir}")
    return output_paths


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
    data: Union[List[Any], np.ndarray], # Allow numpy array
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split data into training, validation, and test sets.

    Args:
        data: Input data (list or numpy array)
        train_ratio: Ratio of data for training set
        val_ratio: Ratio of data for validation set
        test_ratio: Ratio of data for test set
        seed: Random seed for shuffling

    Returns:
        Tuple containing train, validation, and test sets
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Ensure ratios sum to 1
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    # Shuffle indices
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    # Calculate split indices
    train_end = int(train_ratio * len(data))
    val_end = train_end + int(val_ratio * len(data))

    # Split data using shuffled indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Handle numpy array input efficiently
    if isinstance(data, np.ndarray):
        train_data = data[train_indices]
        val_data = data[val_indices]
        test_data = data[test_indices]
    else:
        # Handle list input
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

    return train_data, val_data, test_data 