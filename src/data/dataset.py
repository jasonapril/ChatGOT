"""
Dataset implementations for character-level models.
"""
import torch
from torch.utils.data import Dataset

from .base import TextDataset


class CharDataset(TextDataset):
    """Character-level dataset for transformer models."""
    
    def __init__(self, text, block_size):
        """
        Initialize a character-level dataset.
        
        Args:
            text (str): The training text
            block_size (int): Maximum sequence length
        """
        super().__init__()
        self.text = text
        self.block_size = block_size
        
        # Create character vocabulary from the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        self.data = [self.char_to_idx[c] for c in self.text]
    
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Args:
            idx (int): Index to retrieve
            
        Returns:
            tuple: (x, y) where x is the input sequence and y is the target sequence
        """
        # Get a chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def decode(self, indices):
        """
        Convert indices back to characters.
        
        Args:
            indices (list or tensor): List of indices to convert
            
        Returns:
            str: Decoded text
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.idx_to_char[i] for i in indices])
    
    def get_config(self) -> dict:
        """
        Get the dataset configuration.
        
        Returns:
            Dictionary with the dataset configuration
        """
        config = super().get_config()
        config.update({
            "format": "character",
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
        })
        return config


def load_data(text_path, block_size=1024):
    """
    Load data from a text file into a character dataset.
    
    Args:
        text_path (str): Path to the text file
        block_size (int): Maximum sequence length
        
    Returns:
        CharDataset: The dataset
    """
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return CharDataset(text, block_size) 