"""
Factory functions and DataManager for creating datasets and dataloaders from configurations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type

import torch
from torch.utils.data import DataLoader, Dataset # Added Dataset
import hydra.utils
from omegaconf import DictConfig, OmegaConf
import hydra

# Use relative import as factory.py is in the same package level as base.py
from .base import BaseDataset 
# Remove direct import of specific dataset types unless absolutely needed for factory logic
# from .dataset import CharDataset # Example, adjust if others are needed
from .tokenizers.base import BaseTokenizer # Needed for collate setup
# Import specific collate functions if used directly
# from .collation import character_collate_fn # Commented out - Module not found
from functools import partial

logger = logging.getLogger(__name__)


# --- DataManager Class --- #
class DataManager:
    """
    Data manager for Craft.
    Handles dataset loading, preprocessing, and batch preparation.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the data manager.
        
        Args:
            config: Root data configuration (OmegaConf DictConfig)
        """
        self.config = config
        self.datasets: Dict[str, BaseDataset] = {}
        self.dataloaders: Dict[str, DataLoader] = {}
        self.tokenizer: Optional[BaseTokenizer] = None # Hold tokenizer instance
        
        # Initialize tokenizer if configured
        if "tokenizer" in self.config:
             try:
                 # Assume tokenizer config is nested
                 tokenizer_cfg = self.config.tokenizer
                 self.tokenizer = hydra.utils.instantiate(tokenizer_cfg, _convert_="partial")
                 logger.info(f"Initialized Tokenizer: {type(self.tokenizer).__name__}")
             except Exception as e:
                 logger.error(f"Failed to instantiate tokenizer from config: {e}", exc_info=True)
                 # Decide if this is fatal? Perhaps allow proceeding without tokenizer.
                 self.tokenizer = None

    def get_dataset(self, split: str) -> Optional[BaseDataset]:
        """
        Get or load a dataset for a specific split.
        Uses configuration provided during DataManager initialization.
        
        Args:
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Dataset instance or None if config is missing for the split.
        """
        if split in self.datasets:
            return self.datasets[split]

        # Check if split configuration exists
        if "datasets" not in self.config or split not in self.config.datasets:
             logger.warning(f"No dataset configuration found for split '{split}' in data config.")
             return None

        # Get the config for the specific dataset split
        dataset_cfg = self.config.datasets[split]
        logger.info(f"Loading dataset for split '{split}' with config: {OmegaConf.to_container(dataset_cfg)}")

        try:
            # Instantiate the dataset using Hydra
            # Pass the tokenizer instance if the dataset needs it
            dataset: BaseDataset = hydra.utils.instantiate(dataset_cfg, tokenizer=self.tokenizer, _convert_="partial")
            self.datasets[split] = dataset
            logger.info(f"Successfully loaded dataset for split '{split}': {type(dataset).__name__}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to instantiate dataset for split '{split}' from config: {e}", exc_info=True)
            return None

    def get_dataloader(self, split: str) -> Optional[DataLoader]:
        """
        Get or create a DataLoader for a specific split.

        Args:
            split: Data split ('train', 'val', 'test')
            
        Returns:
            DataLoader instance or None if dataset cannot be loaded.
        """
        if split in self.dataloaders:
            return self.dataloaders[split]
            
        dataset = self.get_dataset(split)
        if dataset is None:
             return None # Dataset couldn't be loaded

        # Determine dataloader parameters from the main config or defaults
        batch_size = self.config.get("batch_size", 32)
        num_workers = self.config.get("num_workers", 0)
        pin_memory = self.config.get("pin_memory", torch.cuda.is_available())
        shuffle = (split == 'train') # Default shuffle only for train
        # Allow overriding shuffle per split if needed from config (e.g., config.datasets.train.shuffle=True)
        shuffle = self.config.datasets.get(split, {}).get("shuffle", shuffle)

        # --- Determine Collate Function --- #
        collate_fn_cfg = self.config.get("collate_fn")
        collate_fn = None
        if collate_fn_cfg:
            logger.info(f"Attempting to instantiate collate_fn from config for split '{split}'")
            try:
                 # Instantiate collate function - potentially passing tokenizer
                 # Use _convert_="partial" if it needs arguments not available yet
                 # Or pass args directly if known (like tokenizer)
                 collate_fn = hydra.utils.instantiate(collate_fn_cfg, tokenizer=self.tokenizer, _convert_="partial")
                 logger.info(f"Using collate_fn: {getattr(collate_fn, '__name__', repr(collate_fn))}")
            except Exception as e:
                 logger.error(f"Failed to instantiate collate_fn from config: {e}. Using default collate.", exc_info=True)
                 collate_fn = None
        elif hasattr(dataset, 'collate_fn') and callable(dataset.collate_fn):
            # Use collate_fn defined on the dataset instance itself (if available)
            # This is useful if the collate function is tightly coupled with the dataset
            collate_fn = dataset.collate_fn 
            logger.info(f"Using collate_fn from dataset instance: {getattr(collate_fn, '__name__', repr(collate_fn))}")
        else:
            logger.info(f"No specific collate_fn configured or found on dataset for split '{split}'. Using default DataLoader collate.")
        # ---------------------------------- #

        try:
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn # Use instantiated or default collate_fn
            )
            self.dataloaders[split] = dataloader
            logger.info(f"Created DataLoader for split '{split}'.")
            return dataloader
        except Exception as e:
            logger.error(f"Failed to create DataLoader for split '{split}': {e}", exc_info=True)
            return None

    def get_tokenizer(self) -> Optional[BaseTokenizer]:
         """Returns the tokenizer instance managed by the DataManager."""
         return self.tokenizer

# --- Factory/Utility Functions --- #

def create_data_manager_from_config(config: DictConfig) -> DataManager:
    """
    Instantiates a DataManager using the provided configuration.

    Args:
        config: The OmegaConf DictConfig object, expected to contain a 'data' section.

    Returns:
        An initialized DataManager instance.

    Raises:
        ValueError: If the configuration does not contain the necessary 'data' section.
    """
    if "data" not in config:
        raise ValueError("Configuration must contain a 'data' section to create DataManager.")
    
    logger.info("Creating DataManager from configuration...")
    # Pass only the data part of the config to the DataManager
    data_manager = DataManager(config.data)
    return data_manager

def prepare_dataloaders_from_config(config: DictConfig) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Optional[BaseTokenizer]]:
    """
    High-level function to create train, validation, and test dataloaders 
    and the tokenizer directly from the main application config.

    Args:
        config: The main OmegaConf DictConfig object.

    Returns:
        A tuple containing (train_loader, val_loader, test_loader, tokenizer).
        Loaders/tokenizer will be None if not configured or if creation fails.
    """
    try:
        data_manager = create_data_manager_from_config(config)
        
        train_loader = data_manager.get_dataloader('train')
        val_loader = data_manager.get_dataloader('val')
        test_loader = data_manager.get_dataloader('test')
        tokenizer = data_manager.get_tokenizer()
        
        return train_loader, val_loader, test_loader, tokenizer
    except ValueError as e:
        logger.error(f"Configuration error during dataloader preparation: {e}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error during dataloader preparation: {e}", exc_info=True)
        return None, None, None, None

# --- Older/Alternative Factory Functions (kept for reference/potential use) --- #

def create_dataset(config: Dict[str, Any]) -> BaseDataset:
    """
    Factory function to create a dataset instance based on configuration.

    Args:
        config: Dictionary containing dataset configuration, including a `_target_` key.

    Returns:
        An instance of a BaseDataset subclass.

    Raises:
        ValueError: If `_target_` key is missing or instantiation fails.
    """
    logger.info(f"Attempting to create dataset from config: {config.get('_target_', 'N/A')}")
    if config is None or not isinstance(config, dict) or "_target_" not in config:
        raise ValueError("Dataset configuration must be a dictionary with a '_target_' key.")
    
    try:
        # Use Hydra utility to instantiate the class specified by _target_
        dataset_instance = hydra.utils.instantiate(config, _convert_="partial")
        if not isinstance(dataset_instance, Dataset):
            raise TypeError(f"Instantiated object is not a PyTorch Dataset: {type(dataset_instance)}")
        logger.info(f"Successfully created dataset: {type(dataset_instance).__name__}")
        # Assuming BaseDataset is the intended type for logging/checks
        if isinstance(dataset_instance, BaseDataset):
            dataset_instance.summary() 
        return dataset_instance
    except Exception as e:
        logger.error(f"Failed to instantiate dataset from config ({config.get('_target_')}): {e}", exc_info=True)
        raise ValueError(f"Could not create dataset from config: {e}") from e

def create_dataloader(
    dataset: Dataset, 
    dataloader_config: Dict[str, Any], 
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    """
    Factory function to create a DataLoader instance.

    Args:
        dataset: The Dataset instance to wrap.
        dataloader_config: Dictionary containing DataLoader configuration 
                          (e.g., batch_size, shuffle, num_workers).
        collate_fn: Optional custom collate function.

    Returns:
        A DataLoader instance.

    Raises:
        ValueError: If configuration is invalid or instantiation fails.
    """
    logger.info(f"Creating DataLoader for dataset: {type(dataset).__name__}")
    if dataloader_config is None or not isinstance(dataloader_config, dict):
        raise ValueError("DataLoader configuration must be a dictionary.")
        
    try:
        # Prepare DataLoader arguments
        loader_args = {
            "dataset": dataset,
            "batch_size": dataloader_config.get("batch_size", 1),
            "shuffle": dataloader_config.get("shuffle", False),
            "num_workers": dataloader_config.get("num_workers", 0),
            "pin_memory": dataloader_config.get("pin_memory", torch.cuda.is_available()),
            "drop_last": dataloader_config.get("drop_last", False),
        }
        if collate_fn:
            loader_args["collate_fn"] = collate_fn
            logger.info(f"Using provided collate function: {getattr(collate_fn, '__name__', repr(collate_fn))}")
        elif hasattr(dataset, 'collate_fn') and callable(getattr(dataset, 'collate_fn')):
             loader_args["collate_fn"] = dataset.collate_fn
             logger.info(f"Using collate_fn from dataset: {getattr(dataset.collate_fn, '__name__', repr(dataset.collate_fn))}")

        dataloader_instance = DataLoader(**loader_args)
        logger.info(f"Successfully created DataLoader with batch size {loader_args['batch_size']}")
        return dataloader_instance
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}", exc_info=True)
        raise ValueError(f"Could not create DataLoader: {e}") from e

def create_data_loaders_from_config(config: DictConfig) -> Dict[str, DataLoader]:
    """
    Creates train, validation, and test DataLoaders based on a Hydra config.
    Assumes a structure like:
    data:
      datasets:
        train:
          _target_: ...
          path: ...
        val:
          ...
        test:
          ...
      dataloader:
        batch_size: ...
        num_workers: ...
      tokenizer:
        _target_: ... 
      collate_fn:
         _target_: ... # Optional collate function config

    Args:
        config: The OmegaConf DictConfig object.

    Returns:
        A dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    dataloaders = {}
    tokenizer = None
    collate_fn = None

    if "data" not in config:
        raise ValueError("Config missing 'data' section.")
    data_cfg = config.data

    # 1. Instantiate Tokenizer (if configured)
    if "tokenizer" in data_cfg:
        try:
            tokenizer = hydra.utils.instantiate(data_cfg.tokenizer, _convert_="partial")
            logger.info(f"Instantiated tokenizer: {type(tokenizer).__name__}")
        except Exception as e:
            logger.error(f"Failed to instantiate tokenizer: {e}. Proceeding without tokenizer.", exc_info=True)
            tokenizer = None
            
    # 2. Instantiate Collate Function (if configured)
    if "collate_fn" in data_cfg:
         try:
             # Pass tokenizer to collate function if it exists
             collate_fn = hydra.utils.instantiate(data_cfg.collate_fn, tokenizer=tokenizer, _convert_="partial")
             logger.info(f"Instantiated collate function: {getattr(collate_fn, '__name__', repr(collate_fn))}")
         except Exception as e:
             logger.error(f"Failed to instantiate collate_fn: {e}. Using default collate where needed.", exc_info=True)
             collate_fn = None

    # 3. Instantiate Datasets and DataLoaders for each split
    if "datasets" not in data_cfg:
        raise ValueError("Config missing 'data.datasets' section.")

    for split in ["train", "val", "test"]:
        if split in data_cfg.datasets:
            logger.info(f"Processing '{split}' split...")
            split_dataset_cfg = data_cfg.datasets[split]
            try:
                # Pass tokenizer to dataset if it exists
                dataset = hydra.utils.instantiate(split_dataset_cfg, tokenizer=tokenizer, _convert_="partial")
                if not isinstance(dataset, Dataset):
                     logger.error(f"Instantiated object for split '{split}' is not a Dataset: {type(dataset)}")
                     continue # Skip this split
                
                # Determine collate_fn for this specific loader
                # Priority: 1) Global collate_fn, 2) Dataset's collate_fn, 3) Default
                current_collate_fn = collate_fn # Use global one if available
                if current_collate_fn is None and hasattr(dataset, 'collate_fn') and callable(getattr(dataset, 'collate_fn')):
                    current_collate_fn = dataset.collate_fn
                    logger.info(f"Using collate_fn defined on '{split}' dataset.")
                elif current_collate_fn is None:
                     logger.info(f"Using default collate for '{split}' dataloader.")

                # Use dataloader config from the top level 'data' section
                dataloader_cfg = data_cfg.get("dataloader", {})
                # Allow split-specific overrides for dataloader args like shuffle
                shuffle_default = (split == 'train')
                loader_args = {
                    "batch_size": dataloader_cfg.get("batch_size", 32),
                    "num_workers": dataloader_cfg.get("num_workers", 0),
                    "pin_memory": dataloader_cfg.get("pin_memory", torch.cuda.is_available()),
                    "drop_last": dataloader_cfg.get("drop_last", False),
                    "shuffle": split_dataset_cfg.get("shuffle", shuffle_default) # Allow override
                }

                dataloaders[split] = DataLoader(
                    dataset=dataset,
                    collate_fn=current_collate_fn,
                    **loader_args
                )
                logger.info(f"Successfully created DataLoader for split '{split}'.")

            except Exception as e:
                logger.error(f"Failed to create dataset or dataloader for split '{split}': {e}", exc_info=True)
        else:
            logger.warning(f"No dataset configuration found for split '{split}'. Skipping.")

    return dataloaders 