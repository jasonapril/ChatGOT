"""
Dataset implementations for character-level models.
"""
import torch
import logging
import os
import json # Import json
from typing import Dict, Any

from torch.utils.data import Dataset

# Import the new BaseDataset directly
from .base import BaseDataset

logger = logging.getLogger(__name__)


class CharDataset(BaseDataset):
    """Character-level dataset from a text file, using precomputed vocabulary."""
    
    # Modified __init__ to accept vocab_path
    def __init__(self, file_path: str, block_size: int, vocab_path: str):
        """
        Initializes the CharDataset using a precomputed vocabulary.
        
        Args:
            file_path (str): Path to the text file containing the data sequence.
            block_size (int): Maximum sequence length for blocks.
            vocab_path (str): Path to the precomputed vocabulary JSON file.
        """
        # Store main parameters
        self.file_path = file_path
        self.block_size = block_size
        self.vocab_path = vocab_path
        
        # --- Load Precomputed Vocabulary --- 
        logger.info(f"Loading precomputed vocabulary from: {self.vocab_path}")
        if not os.path.exists(self.vocab_path):
            logger.error(f"Vocabulary file not found: {self.vocab_path}")
            raise FileNotFoundError(f"Vocabulary file not found: {self.vocab_path}")
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            self.char_to_idx = vocab_data['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()} # Ensure keys are int
            self.vocab_size = vocab_data['vocab_size']
            logger.info(f"Loaded vocabulary. Size: {self.vocab_size}")
        except Exception as e:
            logger.error(f"Failed to load or parse vocabulary file {self.vocab_path}: {e}", exc_info=True)
            raise

        # --- Load and Process Text Data using the loaded vocabulary ---
        logger.info(f"Loading character data from: {self.file_path}")
        if not os.path.exists(self.file_path):
            logger.error(f"Data file path not found: {self.file_path}")
            raise FileNotFoundError(f"Invalid data file path: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.text = f.read()
        except Exception as e:
            logger.error(f"Failed to read data file {self.file_path}: {e}", exc_info=True)
            raise

        if not self.text:
            logger.warning(f"Loaded text file {self.file_path} is empty.")
            self.data = [] # Represent as empty list
        else:
            # Convert text to indices using the loaded char_to_idx map
            # Handle characters potentially not in the loaded vocab (e.g., if data file differs slightly)
            self.data = []
            unknown_chars = set()
            for char in self.text:
                idx = self.char_to_idx.get(char)
                if idx is not None:
                    self.data.append(idx)
                else:
                    # Option 1: Skip unknown characters (simple)
                    # Option 2: Map to a special <UNK> token (requires <UNK> in vocab)
                    # Option 3: Raise an error
                    unknown_chars.add(char)
            if unknown_chars:
                logger.warning(f"Found {len(unknown_chars)} character(s) in {self.file_path} not present in vocabulary {self.vocab_path}. These characters were skipped: {unknown_chars}")
            logger.info(f"Loaded and indexed {len(self.data)} characters from {self.file_path} using vocabulary from {self.vocab_path}.")
    
    # __len__ remains the same but needs to handle empty data
    def __len__(self):
        """Return the number of possible sequences."""
        if not self.data or len(self.data) <= self.block_size:
            return 0
        return len(self.data) - self.block_size
    
    # Modified __getitem__ to return a dictionary
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample as a dictionary.
        
        Args:
            idx (int): Index to retrieve.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'input_ids' and 'labels'.
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")
            
        # Get a chunk of data (sequence + next char)
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {'input_ids': x, 'labels': y}
    
    # decode method remains useful for this specific dataset type
    def decode(self, indices):
        """
        Convert indices back to characters.
        
        Args:
            indices (list or tensor): List of indices to convert.
            
        Returns:
            str: Decoded text.
        """
        if not self.idx_to_char: # Handle empty dataset case
            return ""
            
        if isinstance(indices, torch.Tensor):
            # Ensure tensor is on CPU and convert to list
            indices = indices.cpu().tolist()
            
        # Handle potential nested lists if batch dimension wasn't squeezed
        if indices and isinstance(indices[0], list):
             indices = indices[0] # Assume first item if nested

        return ''.join([self.idx_to_char.get(i, '?') for i in indices]) # Use .get for safety
    
    # Removed get_config method, base class handles config storage.

# Removed load_data function. Use factory functions like create_dataset_from_config. 