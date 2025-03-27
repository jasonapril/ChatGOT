"""
Base dataset classes and abstractions for Craft.

This module defines base classes and utilities for dataset handling.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base dataset class for Craft.
    
    All dataset implementations should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.data = None
        self._validate_config()
        
    def _validate_config(self) -> None:
        """
        Validate the dataset configuration.
        
        Raises:
            ValueError: If required configuration options are missing
        """
        required_keys = ['paths', 'data']
        missing_keys = [key for key in required_keys if key not in self.config]
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseDataset':
        """
        Create a dataset from a configuration.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Dataset instance
        """
        return cls(config)


class DataManager:
    """
    Data manager for Craft.
    
    Handles dataset loading, preprocessing, and batch preparation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data manager.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.datasets = {}
        self.dataloaders = {}
        
    def load_dataset(self, name: str, dataset_cls: type, split: str = 'train') -> BaseDataset:
        """
        Load a dataset.
        
        Args:
            name: Dataset name
            dataset_cls: Dataset class
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Dataset instance
        """
        # Create a config specific to this split
        split_config = self.config.copy()
        if 'splits' in self.config and split in self.config['splits']:
            # Update with split-specific config
            split_config.update(self.config['splits'][split])
        
        # Create the dataset
        dataset = dataset_cls.from_config(split_config)
        
        # Store the dataset
        key = f"{name}_{split}"
        self.datasets[key] = dataset
        
        return dataset
    
    def create_dataloader(
        self, 
        name: str, 
        split: str = 'train', 
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """
        Create a data loader for a dataset.
        
        Args:
            name: Dataset name
            split: Data split ('train', 'val', 'test')
            batch_size: Batch size (overrides config)
            shuffle: Whether to shuffle the data (overrides config)
            num_workers: Number of worker processes (overrides config)
            collate_fn: Function to collate samples into batches
            
        Returns:
            DataLoader instance
        """
        key = f"{name}_{split}"
        
        if key not in self.datasets:
            raise ValueError(f"Dataset {key} not loaded")
        
        dataset = self.datasets[key]
        
        # Get dataloader parameters from config or use provided values
        dl_params = {
            'batch_size': batch_size or self.config.get('batch_size', 32),
            'num_workers': num_workers or self.config.get('num_workers', 0),
            'pin_memory': self.config.get('pin_memory', True)
        }
        
        # Set shuffle based on split if not provided
        if shuffle is None:
            dl_params['shuffle'] = (split == 'train')
        else:
            dl_params['shuffle'] = shuffle
        
        # Set collate function if provided
        if collate_fn:
            dl_params['collate_fn'] = collate_fn
        
        # Create and store the dataloader
        dataloader = DataLoader(dataset, **dl_params)
        self.dataloaders[key] = dataloader
        
        return dataloader
    
    def get_dataloader(self, name: str, split: str = 'train') -> DataLoader:
        """
        Get a data loader.
        
        Args:
            name: Dataset name
            split: Data split ('train', 'val', 'test')
            
        Returns:
            DataLoader instance
        """
        key = f"{name}_{split}"
        
        if key not in self.dataloaders:
            raise ValueError(f"DataLoader {key} not created")
        
        return self.dataloaders[key]


def create_data_manager(config: Dict[str, Any]) -> DataManager:
    """
    Create a data manager from a configuration.
    
    Args:
        config: Data configuration
        
    Returns:
        DataManager instance
    """
    return DataManager(config)


class TextDataset(BaseDataset):
    """
    Abstract base class for text datasets in Craft.
    
    This class extends BaseDataset with text-specific functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the text dataset."""
        super().__init__(config)
        self.data_type = "text"
    
    def decode(self, indices):
        """
        Convert indices back to text.
        
        Args:
            indices: Indices to convert
            
        Returns:
            Decoded text
        """
        pass


class ImageDataset(BaseDataset):
    """
    Abstract base class for image datasets in Craft.
    
    This class extends BaseDataset with image-specific functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the image dataset."""
        super().__init__(config)
        self.data_type = "image"


class MultiModalDataset(BaseDataset):
    """
    Abstract base class for multi-modal datasets in Craft.
    
    This class extends BaseDataset with multi-modal functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the multi-modal dataset."""
        super().__init__(config)
        self.data_type = "multi-modal"


def create_dataloaders(
    dataset: BaseDataset,
    batch_size: int,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders from a dataset.
    
    Args:
        dataset: The dataset to split
        batch_size: Batch size for the dataloaders
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        pin_memory: Whether to use pinned memory
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Calculate sizes
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Split dataset
    if val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
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
    else:
        # Create only training dataloader
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        return train_dataloader, None


def create_dataset_from_config(config: Dict) -> BaseDataset:
    """
    Create a dataset from a configuration dictionary.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        Instantiated dataset
    """
    data_type = config.get("data_type", "text")
    
    if data_type == "text":
        from .dataset import CharDataset
        
        if config.get("format") == "character":
            # Load text from file
            with open(config["data_path"], 'r', encoding='utf-8') as f:
                text = f.read()
                
            return CharDataset(text, config.get("block_size", 1024))
        else:
            raise ValueError(f"Unknown text format: {config.get('format')}")
    elif data_type == "image":
        raise NotImplementedError("Image datasets not yet implemented")
    elif data_type == "multi-modal":
        raise NotImplementedError("Multi-modal datasets not yet implemented")
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def prepare_dataloaders_from_config(config: Dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare dataloaders from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create dataset
    dataset = create_dataset_from_config(config["data"])
    
    # Create dataloaders
    return create_dataloaders(
        dataset=dataset,
        batch_size=config["data"].get("batch_size", 32),
        val_split=config["data"].get("val_split", 0.1),
        seed=config.get("seed", 42),
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    ) 