"""
Dataset implementations for character-level models.
"""
import torch
import logging
import os
import json # Import json
import pickle
import numpy as np
from typing import Dict, Any

from torch.utils.data import Dataset

# Import the new BaseDataset directly
from .base import BaseDataset

logger = logging.getLogger(__name__)


# Removed load_data function. Use factory functions like create_dataset_from_config. 


class PickledDataset(BaseDataset):
    """Dataset that loads pre-tokenized data from a pickle file."""

    def __init__(self, file_path: str, block_size: int, vocab_path: str = None):
        """
        Initializes the Dataset from a .pkl file containing tokenized data.

        Args:
            file_path (str): Path to the .pkl file.
            block_size (int): Maximum sequence length for blocks.
            vocab_path (str, optional): Path to a JSON file containing vocabulary data.
        """
        self.file_path = file_path
        self.block_size = block_size

        # Load vocabulary data if provided
        if vocab_path:
            logger.info(f"Loading vocabulary from: {vocab_path}")
            if not os.path.exists(vocab_path):
                logger.error(f"Vocabulary file not found: {vocab_path}")
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
            
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                
                # Convert string keys to integers for idx_to_char
                self.char_to_idx = vocab_data.get('char_to_idx', {})
                self.idx_to_char = {int(k): v for k, v in vocab_data.get('idx_to_char', {}).items()}
                self.vocab_size = vocab_data.get('vocab_size', len(self.char_to_idx))
                
                logger.info(f"Loaded vocabulary with size {self.vocab_size}")
            except Exception as e:
                logger.error(f"Failed to load vocabulary file {vocab_path}: {e}", exc_info=True)
                raise

        logger.info(f"Loading pre-tokenized data from: {self.file_path}")
        if not os.path.exists(self.file_path):
            logger.error(f"Pickled data file not found: {self.file_path}")
            raise FileNotFoundError(f"Pickled data file not found: {self.file_path}")
        
        try:
            with open(self.file_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Load token IDs and other data
            self.token_ids = data_dict.get('token_ids', [])
            if not vocab_path:
                self.char_to_idx = data_dict.get('char_to_idx', {})
                self.idx_to_char = {int(k): v for k, v in data_dict.get('idx_to_char', {}).items()}
                self.vocab_size = data_dict.get('vocab_size', len(self.char_to_idx))

            logger.info(f"Loaded {len(self.token_ids)} tokens with vocabulary size {self.vocab_size}")

        except Exception as e:
            logger.error(f"Failed to load or parse pickle file {self.file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        """Return the length of the dataset."""
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx):
        """Get a block of token IDs."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Get block of token IDs
        x = self.token_ids[idx:idx + self.block_size]
        y = self.token_ids[idx + 1:idx + self.block_size + 1]

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def decode(self, tokens):
        """Decode a sequence of token IDs back to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ''.join(self.idx_to_char[int(idx)] for idx in tokens) 