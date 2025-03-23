#!/usr/bin/env python
"""
Data Processor Module
====================

This module handles the conversion of raw text data into processed format
for training the ChatGoT model. It includes functions for:

1. Character-level tokenization
2. Creating training sequences with proper context length
3. Saving processed data for reuse in training
"""

import os
import pickle
import logging
import torch
import re
from typing import Dict, List, Tuple, Any, Optional

def read_raw_file(file_path: str) -> str:
    """
    Read raw text file.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Raw text content
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    return text

def create_character_mappings(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character to index and index to character mappings.
    
    Args:
        text: Raw text content
        
    Returns:
        Tuple of (char_to_idx, idx_to_char)
    """
    # Get unique characters
    chars = sorted(list(set(text)))
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    logging.info(f"Created vocabulary with {len(chars)} unique characters")
    return char_to_idx, idx_to_char

def create_training_sequences(
    text: str, 
    char_to_idx: Dict[str, int], 
    sequence_length: int = 1024,
    stride: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create training sequences from raw text.
    
    Args:
        text: Raw text content
        char_to_idx: Character to index mapping
        sequence_length: Length of each sequence
        stride: Stride between sequences (default: sequence_length//2)
        
    Returns:
        List of (input, target) tensors
    """
    if stride is None:
        stride = max(1, sequence_length // 2)  # 50% overlap by default
    
    # Convert text to indices
    indices = [char_to_idx[ch] for ch in text]
    
    # Create sequences
    sequences = []
    for i in range(0, len(indices) - sequence_length, stride):
        # Input is sequence of chars
        input_seq = torch.tensor(indices[i:i + sequence_length], dtype=torch.long)
        
        # Target is sequence shifted by 1
        target_seq = torch.tensor(indices[i + 1:i + sequence_length + 1], dtype=torch.long)
        
        sequences.append((input_seq, target_seq))
    
    logging.info(f"Created {len(sequences)} sequences of length {sequence_length}")
    return sequences

def apply_text_preprocessing(text: str) -> str:
    """
    Apply preprocessing steps to the raw text.
    
    Args:
        text: Raw text content
        
    Returns:
        Preprocessed text
    """
    # Replace multiple consecutive newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Remove any non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    return text

def split_train_val(
    sequences: List[Tuple[torch.Tensor, torch.Tensor]], 
    val_split: float = 0.1
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Split sequences into training and validation sets.
    
    Args:
        sequences: List of sequence tuples
        val_split: Validation split ratio (0-1)
        
    Returns:
        Tuple of (train_sequences, val_sequences)
    """
    val_size = int(len(sequences) * val_split)
    train_sequences = sequences[:-val_size] if val_size > 0 else sequences
    val_sequences = sequences[-val_size:] if val_size > 0 else []
    
    logging.info(f"Split data into {len(train_sequences)} training and {len(val_sequences)} validation sequences")
    return train_sequences, val_sequences

def process_raw_data(
    input_file: str, 
    output_path: str, 
    sequence_length: int = 1024,
    val_split: float = 0.1,
    stride: Optional[int] = None
) -> None:
    """
    Process raw text data into format ready for training.
    
    Args:
        input_file: Path to raw text file
        output_path: Path to save processed data
        sequence_length: Length of each sequence
        val_split: Validation split ratio
        stride: Stride between sequences
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logging.info(f"Processing raw data from: {input_file}")
    logging.info(f"Sequence length: {sequence_length}")
    
    # Read and preprocess text
    text = read_raw_file(input_file)
    text = apply_text_preprocessing(text)
    
    logging.info(f"Raw text size: {len(text)} characters")
    
    # Create character mappings
    char_to_idx, idx_to_char = create_character_mappings(text)
    
    # Create training sequences
    sequences = create_training_sequences(
        text, 
        char_to_idx, 
        sequence_length=sequence_length,
        stride=stride
    )
    
    # Split into training and validation
    train_sequences, val_sequences = split_train_val(sequences, val_split)
    
    # Package data
    data = {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'sequence_length': sequence_length,
        'metadata': {
            'original_file': input_file,
            'text_length': len(text),
            'vocab_size': len(char_to_idx),
            'processed_at': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
    }
    
    # Save data
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"Processed data saved to: {output_path}")
    logging.info(f"Total sequences: {len(sequences)}")
    logging.info(f"Vocabulary size: {len(char_to_idx)}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process raw text data for training")
    parser.add_argument("input_file", type=str, help="Path to raw text file")
    parser.add_argument("--output_path", type=str, default="processed_data/processed_data.pkl",
                       help="Path to save processed data")
    parser.add_argument("--sequence_length", type=int, default=1024,
                       help="Length of each sequence")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio (0-1)")
    parser.add_argument("--stride", type=int, default=None,
                       help="Stride between sequences (default: sequence_length//2)")
    
    args = parser.parse_args()
    
    # Process data
    process_raw_data(
        args.input_file,
        args.output_path,
        args.sequence_length,
        args.val_split,
        args.stride
    ) 