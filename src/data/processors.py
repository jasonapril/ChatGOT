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
    Prepare text data for training. Supports character-level or pre-trained tokenizers.
    
    Args:
        input_file: Path to the input text file
        output_dir: Directory to save processed data
        config: Configuration dictionary (expected to have data.tokenizer_name)
        
    Returns:
        Path to the main processed dataset file (e.g., train split path)
        Note: This might need adjustment when splitting is fully implemented.
    """
    logging.info(f"Processing text data from {input_file}")
    
    # Get configuration
    cfg = config if config is not None else {}
    data_cfg = cfg.get("data", {})
    
    # Determine tokenizer type/name
    tokenizer_name = data_cfg.get("tokenizer_name", None)
    text_format = data_cfg.get("format", "character") # Fallback if no tokenizer

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    logging.info(f"Read {len(text)} characters from {input_file}")
    
    processed_data = {}
    vocab_size = None
    tokenizer = None

    # Process based on tokenizer or format
    if tokenizer_name:
        logging.info(f"Using tokenizer: {tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Add special tokens if they don't exist? Often handled by dataset/model.
            # Example: if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Tokenize the entire text
            # Use truncation/padding later in dataset loading if needed
            # For large files, consider processing in chunks or using datasets library
            logging.info(f"Tokenizing text with {tokenizer_name}...")
            # Note: encode adds special tokens by default depending on tokenizer type
            token_ids = tokenizer.encode(text) 
            logging.info(f"Tokenized into {len(token_ids)} tokens.")

            # Store tokenized data and tokenizer info
            # Using numpy for potential memory mapping later
            processed_data["token_ids"] = np.array(token_ids, dtype=np.uint16) # Use efficient dtype
            processed_data["tokenizer_name"] = tokenizer_name
            vocab_size = tokenizer.vocab_size
            processed_data["vocab_size"] = vocab_size
            # Save necessary tokenizer files alongside? Or assume name is enough?
            # For now, assume name is sufficient for reloading.

        except Exception as e:
            logging.error(f"Failed to load or use tokenizer '{tokenizer_name}': {e}", exc_info=True)
            raise

    elif text_format == "character":
        # Character-level processing (existing logic)
        logging.info("Using character-level tokenization.")
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        
        # Store original text and mappings
        processed_data = {
            "text": text, # Keep original text for CharDataset
            "chars": chars,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "vocab_size": vocab_size
        }
        logging.info(f"Vocabulary size: {vocab_size} unique characters")
    else:
        raise ValueError(f"Unsupported text format/tokenizer: format='{text_format}', tokenizer_name='{tokenizer_name}'")
    
    # --- Integrate Data Splitting --- 
    # Get split ratios from config
    # Default to 80/10/10 if not provided
    split_ratios = data_cfg.get('split_ratios', [0.8, 0.1, 0.1])
    if len(split_ratios) != 3 or not all(isinstance(x, (int, float)) for x in split_ratios) or abs(sum(split_ratios) - 1.0) > 1e-6:
        logging.warning(f"Invalid split_ratios '{split_ratios}'. Must be 3 numbers summing to 1. Using default [0.8, 0.1, 0.1].")
        split_ratios = [0.8, 0.1, 0.1]
    train_ratio, val_ratio, test_ratio = split_ratios
    seed = cfg.get("seed", 42) # Use the global seed

    # Decide what data to split
    data_to_split = None
    if tokenizer_name and "token_ids" in processed_data:
        data_to_split = processed_data["token_ids"]
        logging.info(f"Splitting {len(data_to_split)} token IDs...")
    elif text_format == "character" and "text" in processed_data:
        # Note: Splitting raw text might break context across splits.
        # Better approach for char level might be to tokenize first, then split IDs.
        # For now, matching previous potential behavior but logging warning.
        # Let's tokenize first, then split IDs for character level too.
        char_to_idx = processed_data["char_to_idx"]
        raw_text = processed_data["text"]
        token_ids = [char_to_idx.get(c, 0) for c in raw_text] # Simple tokenization
        data_to_split = np.array(token_ids, dtype=np.uint16)
        # Remove raw text if we split IDs
        # processed_data.pop("text", None)
        logging.info(f"Splitting {len(data_to_split)} character token IDs...")
        
    if data_to_split is None or len(data_to_split) == 0:
         logging.error("No data available to split.")
         # Return something reasonable or raise error? Return None for now.
         return None 

    # Split the data (indices)
    # Ensure split_data can handle numpy array directly or list of indices
    # Convert numpy array to list for split_data if needed, though numpy indexing is better
    num_items = len(data_to_split)
    indices = np.arange(num_items)
    train_indices, val_indices, test_indices = split_data(indices, train_ratio, val_ratio, test_ratio, seed)
    
    logging.info(f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Prepare data for each split
    output_paths = {}
    output_filename_base = os.path.splitext(os.path.basename(input_file))[0]
    
    for split_name, split_indices in zip(["train", "val", "test"], [train_indices, val_indices, test_indices]):
        if len(split_indices) == 0:
             logging.info(f"Skipping empty split: {split_name}")
             continue

        split_data_content = {} 
        if tokenizer_name:
            # Save token IDs for the split
            split_data_content["token_ids"] = data_to_split[split_indices]
            split_data_content["tokenizer_name"] = tokenizer_name
            split_data_content["vocab_size"] = vocab_size
        elif text_format == "character":
            # Save token IDs and mappings for the split
            split_data_content["token_ids"] = data_to_split[split_indices]
            split_data_content["chars"] = processed_data["chars"]
            split_data_content["char_to_idx"] = processed_data["char_to_idx"]
            split_data_content["idx_to_char"] = processed_data["idx_to_char"]
            split_data_content["vocab_size"] = vocab_size

        # Save the split data
        split_output_path = os.path.join(output_dir, f"{output_filename_base}_{split_name}.pkl")
        with open(split_output_path, "wb") as f:
            pickle.dump(split_data_content, f)
        logging.info(f"Saved {split_name} split ({len(split_indices)} items/tokens) to {split_output_path}")
        output_paths[split_name] = split_output_path

    # --- End Data Splitting --- 

    # Return the dictionary of output paths
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