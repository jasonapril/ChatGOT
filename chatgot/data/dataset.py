"""Dataset module for character-level text datasets."""
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, random_split

from chatgot.core.config import get_full_path
from chatgot.utils.logging import get_logger

logger = get_logger(__name__)


class CharacterTextDataset(Dataset):
    """Character-level text dataset for training language models."""
    
    def __init__(
        self,
        text: str,
        sequence_length: int = 1024,
        vocab_size: int = 256,
        return_tensors: bool = True,
    ):
        """
        Initialize a character-level text dataset.
        
        Args:
            text: Input text
            sequence_length: Sequence length for each example
            vocab_size: Vocabulary size (limited to 256 for character-level)
            return_tensors: Whether to return PyTorch tensors or lists
        """
        self.text = text
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.return_tensors = return_tensors
        
        # For character-level models, we use the ASCII/Unicode values
        self.data = [ord(c) % self.vocab_size for c in self.text]
        
        logger.info(f"Created dataset with {len(self.data)} characters")
        logger.info(f"Sequence length: {sequence_length}")
        
    def __len__(self) -> int:
        """Get the number of available sequences."""
        # We allow overlapping sequences with a stride of 1
        # and remove any partial sequences at the end
        return max(0, len(self.data) - self.sequence_length)
    
    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, List[int]], Union[torch.Tensor, List[int]]]:
        """
        Get a sequence and its target (next character prediction).
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (input sequence, target sequence)
        """
        # Get input sequence
        input_sequence = self.data[idx:idx + self.sequence_length]
        
        # Target sequence is the same as input but shifted by 1
        # (predict the next character at each position)
        target_sequence = self.data[idx + 1:idx + self.sequence_length + 1]
        
        # Convert to tensors if requested
        if self.return_tensors:
            input_sequence = torch.tensor(input_sequence, dtype=torch.long)
            target_sequence = torch.tensor(target_sequence, dtype=torch.long)
        
        return input_sequence, target_sequence


def load_text_from_file(file_path: Union[str, Path]) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Text content
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    logger.info(f"Loading text from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    logger.info(f"Loaded {len(text)} characters from {file_path}")
    return text


def create_dataloaders(
    text: str,
    sequence_length: int,
    batch_size: int,
    split_ratio: float = 0.9,
    vocab_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders from text.
    
    Args:
        text: Input text
        sequence_length: Sequence length for each example
        batch_size: Batch size
        split_ratio: Ratio of training data (0-1)
        vocab_size: Vocabulary size
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster data transfer to GPU
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create dataset
    dataset = CharacterTextDataset(
        text=text,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        return_tensors=True,
    )
    
    # Split dataset
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training set: {len(train_dataset)} sequences")
    logger.info(f"Validation set: {len(val_dataset)} sequences")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader


def prepare_dataloaders_from_config(
    cfg: DictConfig,
    cache_processed: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare dataloaders based on configuration.
    
    Args:
        cfg: Configuration
        cache_processed: Whether to cache processed data
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Extract parameters from config
    data_cfg = cfg.data
    batch_size = cfg.training.batch_size
    sequence_length = cfg.training.sequence_length
    split_ratio = data_cfg.split_ratio
    num_workers = cfg.system.num_workers
    pin_memory = cfg.system.pin_memory
    seed = cfg.system.seed
    
    # Get file paths
    data_path = get_full_path(cfg.paths.data_file, cfg)
    processed_cache_path = get_full_path(cfg.paths.processed_data, cfg)
    
    # Check if processed cache exists and should be used
    if cache_processed and os.path.exists(processed_cache_path) and not data_cfg.get("refresh", False):
        logger.info(f"Loading processed data from cache: {processed_cache_path}")
        with open(processed_cache_path, "rb") as f:
            processed_data = pickle.load(f)
        
        text = processed_data["text"]
    else:
        # Load and process text
        text = load_text_from_file(data_path)
        
        # Apply text transformations based on config
        if data_cfg.lowercase:
            logger.info("Converting text to lowercase")
            text = text.lower()
            
        if data_cfg.remove_special_chars:
            logger.info("Removing special characters")
            import re
            text = re.sub(r'[^\w\s]', '', text)
        
        # Save processed data to cache if requested
        if cache_processed:
            logger.info(f"Saving processed data to cache: {processed_cache_path}")
            
            processed_data = {
                "text": text,
                "config": {
                    "lowercase": data_cfg.lowercase,
                    "remove_special_chars": data_cfg.remove_special_chars,
                }
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(processed_cache_path), exist_ok=True)
            
            with open(processed_cache_path, "wb") as f:
                pickle.dump(processed_data, f)
    
    # Create dataloaders
    return create_dataloaders(
        text=text,
        sequence_length=sequence_length,
        batch_size=batch_size,
        split_ratio=split_ratio,
        vocab_size=cfg.models.vocab_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    ) 