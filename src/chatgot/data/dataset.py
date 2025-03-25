"""Dataset module for character-level text data."""
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


class CharacterDataset(Dataset):
    """Character-level text dataset for training language models."""
    
    def __init__(
        self,
        sequences: List[Tuple[torch.Tensor, torch.Tensor]],
        char_to_idx: Dict[str, int],
        idx_to_char: Dict[int, str],
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of (input sequence, target sequence) tuples
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
        """
        self.sequences = sequences
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (input sequence, target sequence)
        """
        return self.sequences[idx]
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.char_to_idx)
    
    def decode(self, indices: Union[torch.Tensor, List[int]]) -> str:
        """
        Decode indices to text.
        
        Args:
            indices: Indices to decode
            
        Returns:
            Decoded text
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        return "".join(self.idx_to_char[idx] for idx in indices)


def prepare_dataloaders_from_config(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation dataloaders from config."""
    # Extract parameters from config
    data_cfg = cfg.data
    batch_size = data_cfg.batch_size
    num_workers = data_cfg.num_workers
    pin_memory = data_cfg.pin_memory
    
    # Load processed data
    data_file = Path(cfg.paths.processed_data)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(data_file, "rb") as f:
        processed_data = pickle.load(f)
    
    # Create datasets
    train_dataset = CharacterDataset(
        sequences=processed_data["train_sequences"],
        char_to_idx=processed_data["char_to_idx"],
        idx_to_char=processed_data["idx_to_char"],
    )
    
    val_dataset = CharacterDataset(
        sequences=processed_data["val_sequences"],
        char_to_idx=processed_data["char_to_idx"],
        idx_to_char=processed_data["idx_to_char"],
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_dataloader, val_dataloader 