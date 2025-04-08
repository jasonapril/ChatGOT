"""
Base dataset classes and abstractions for Craft.

This module defines base classes and utilities for dataset handling.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """
    Abstract Base Class for all datasets in this project.
    
    Inherits from torch.utils.data.Dataset and defines a common interface
    that all specific dataset implementations should follow. This promotes
    consistency and adaptability.
    
    Subclasses must implement __len__ and __getitem__. They might also
    implement specific preprocessing, tokenization, or data loading logic
    as needed for their specific data source and task.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseDataset.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary specific 
                                              to the dataset (e.g., file paths, 
                                              preprocessing flags). Defaults to None.
        """
        super().__init__() # Initialize torch.utils.data.Dataset
        self.config = config if config is not None else {}
        logging.info(f"Initializing {self.__class__.__name__} with config: {self.config}")
        # Subclasses should handle their own specific config validation
        
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement __len__")
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Retrieves the sample corresponding to the given index.
        
        The exact format of the returned sample depends on the specific dataset
        and the requirements of the model/training loop (e.g., a dictionary 
        containing 'input_ids', 'labels', etc.).
        
        This method must be implemented by all subclasses.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            Any: The data sample at the specified index.
        """
        raise NotImplementedError("Subclasses must implement __getitem__")
    
    def preprocess(self, sample: Any) -> Any:
        """
        Optional method for applying common preprocessing steps to a sample.
        
        Subclasses can override this method to implement dataset-specific 
        preprocessing like tokenization, normalization, augmentation, etc.
        By default, it returns the sample unchanged.
        
        Args:
            sample (Any): The raw sample retrieved by __getitem__ (or an
                          intermediate stage).
                          
        Returns:
            Any: The processed sample.
        """
        return sample

    def summary(self) -> None:
        """Prints a basic summary of the dataset."""
        try:
            length = len(self)
            logging.info(f"Dataset: {self.__class__.__name__}, Length: {length}")
            if self.config:
                 logging.info(f"Config highlights: { {k: v for k, v in self.config.items() if k in ['data_path', 'split', 'tokenizer_name']} }")
        except NotImplementedError:
            logging.warning(f"Dataset: {self.__class__.__name__} - __len__ not implemented yet.")
        except Exception as e:
            logging.error(f"Error generating summary for {self.__class__.__name__}: {e}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseDataset':
        """
        Create a dataset from a configuration.
        This is a simple helper; more complex instantiation might live in factory.py.
        
        Args:
            config: Dataset configuration
            
        Returns:
            Dataset instance
        """
        # Basic instantiation - assumes config is just keyword args for the class
        # For Hydra instantiation (_target_), use the factory functions.
        return cls(**config) 

# --- DataManager Class Removed --- #
# // ... DataManager definition removed ...

# --- Factory/Utility Functions Removed --- #
# // ... create_data_manager removed ...
# // ... create_dataloaders removed ...
# // ... prepare_dataloaders_from_config removed ...
# // ... create_data_manager_from_config removed ...
# // ... create_data_loaders_from_config removed ...


# Ensure this function is imported where needed, potentially in src/craft/data/__init__.py