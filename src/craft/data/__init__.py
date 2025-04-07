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
from .char_processor import process_char_data
# Import tokenizer base and specific implementations
from .tokenizers.base import Tokenizer
from .tokenizers.char import CharTokenizer
from .tokenizers.sentencepiece import SentencePieceTokenizer
from .tokenizers.subword import SubwordTokenizer

# Import helper(s) moved to utils
from .utils import create_dataloader

# Imports from deleted files removed:
# from .manager import DataManager, create_data_manager_from_config
# from .preparation import prepare_dataloaders_from_config
# from .creation import create_dataset #, create_dataloader
# from .dataset import TextDataset, PickledDataset

# Define what gets imported with "from craft.data import *"
__all__ = [
    # Base Classes
    'BaseDataset',
    'Tokenizer',

    # Specific Tokenizers (optional to expose here)
    'CharTokenizer',
    'SentencePieceTokenizer',
    'SubwordTokenizer',

    # Utility Functions
    'process_char_data',
    'create_dataloader',

    # Type Aliases
    "DataLoadersDict",

    # Removed obsolete/moved items:
    # 'DataManager',
    # 'create_dataset',
    # 'create_data_manager_from_config',
    # 'prepare_dataloaders_from_config',
    # 'TextDataset',
    # 'PickledDataset',
] 