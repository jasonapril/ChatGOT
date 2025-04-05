"""
Data loading and processing utilities for Craft.

Exposes key classes and functions for dataset handling.
"""
from typing import Dict
from torch.utils.data import DataLoader

# Define Type Alias here to avoid circular imports
DataLoadersDict = Dict[str, DataLoader]

# Import key classes and functions to be available directly under craft.data
from .base import BaseDataset
# Import factory functions from the new location
from .factory import (
    create_dataset,
    create_data_loaders_from_config, 
    create_data_manager_from_config,
    prepare_dataloaders_from_config,
    DataManager
)

# Import specific dataset implementations if they are meant to be directly accessible
# Example: from .dataset import CharDataset

# Define what gets imported with "from craft.data import *"
__all__ = [
    'BaseDataset',
    'DataManager',
    'create_dataset',
    'create_data_loaders_from_config',
    'create_data_manager_from_config',
    'prepare_dataloaders_from_config',
    # Add other exported classes/functions like specific datasets if needed
    # 'CharDataset',
    "DataLoadersDict",
    # "TextFileDataset",
    # "prepare_data",
    # "BaseTokenizer",
    # "CharTokenizer",
    # "SubwordTokenizer",
] 