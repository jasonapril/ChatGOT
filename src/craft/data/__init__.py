"""
Data module initialization.
Exposes key components for data loading and processing.
"""
from typing import Dict
from torch.utils.data import DataLoader

# Define Type Alias here to avoid circular imports
DataLoadersDict = Dict[str, DataLoader]

from .base import BaseDataset, create_dataset_from_config, create_data_loaders_from_config
from .dataset import PickledDataset
# from .processors import prepare_data # Example import if needed
# from .tokenizers import BaseTokenizer, CharTokenizer, SubwordTokenizer # Example imports if needed

__all__ = [
    "BaseDataset",
    "PickledDataset",
    # "TextFileDataset",
    "create_dataset_from_config",
    "create_data_loaders_from_config", # Added new factory function
    "DataLoadersDict", # Keep in __all__
    # "prepare_data",
    # "BaseTokenizer",
    # "CharTokenizer",
    # "SubwordTokenizer",
] 