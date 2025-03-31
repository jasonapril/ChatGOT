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

    def __init__(self, file_path: str, block_size: int):
        """
        Initializes the Dataset from a .pkl file containing tokenized data.

        Args:
            file_path (str): Path to the .pkl file.
            block_size (int): Maximum sequence length for blocks.
        """
        self.file_path = file_path
        self.block_size = block_size

        logger.info(f"Loading pre-tokenized data from: {self.file_path}")
        if not os.path.exists(self.file_path):
            logger.error(f"Pickled data file not found: {self.file_path}")
            raise FileNotFoundError(f"Pickled data file not found: {self.file_path}")
        
        try:
            with open(self.file_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            if "token_ids" not in data_dict:
                raise ValueError(f"Pickle file {self.file_path} missing 'token_ids' key.")
            if "vocab_size" not in data_dict:
                 raise ValueError(f"Pickle file {self.file_path} missing 'vocab_size' key.")
                
            # Store token ids (convert to tensor for efficiency if large? Keep as list/numpy for now)
            self.data = data_dict["token_ids"]
            if isinstance(self.data, np.ndarray):
                 # Convert numpy to list of ints, might be safer for slicing?
                 # Or convert to torch tensor later in getitem? Let's keep numpy for now.
                 pass 
            elif not isinstance(self.data, (list, np.ndarray)):
                 raise TypeError(f"'token_ids' in {self.file_path} must be a list or numpy array.")

            self.vocab_size = data_dict["vocab_size"]
            
            # Store other metadata if needed (e.g., tokenizer_name, char maps for char-level)
            self.tokenizer_name = data_dict.get("tokenizer_name")
            self.char_to_idx = data_dict.get("char_to_idx")
            self.idx_to_char = data_dict.get("idx_to_char")

            logger.info(f"Loaded {len(self.data)} tokens. Vocab size: {self.vocab_size}. Tokenizer: {self.tokenizer_name or 'character'}")

        except Exception as e:
            logger.error(f"Failed to load or parse pickle file {self.file_path}: {e}", exc_info=True)
            raise

    def __len__(self):
        """Return the number of sequences (blocks) in the dataset."""
        # Check if data exists and has enough elements for at least one block
        if not hasattr(self, 'data') or self.data.size == 0 or len(self.data) <= self.block_size:
            return 0
        # Calculate the number of full blocks available
        # Subtract block_size because the last block needs a target
        return (len(self.data) - self.block_size) // self.block_size

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

        # Get a chunk of token_ids
        # Slicing works on lists and numpy arrays
        chunk = self.data[idx * self.block_size:idx * self.block_size + self.block_size + 1]
        
        # Convert chunk to tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {'input_ids': x, 'labels': y}
    
    # Optional: Add a decode method if idx_to_char is available
    def decode(self, indices):
        if not hasattr(self, 'idx_to_char') or not self.idx_to_char:
             logger.warning("Decode called but idx_to_char mapping not found in loaded data.")
             return ""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()
        # Handle potential nested lists if batch dim wasn't squeezed
        if indices and isinstance(indices[0], list):
             indices = indices[0] 
        return ''.join([self.idx_to_char.get(i, '?') for i in indices]) 