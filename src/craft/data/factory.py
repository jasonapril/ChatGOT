"""
Factory functions and DataManager for creating datasets and dataloaders from configurations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Type

import torch
from torch.utils.data import DataLoader, Dataset, default_collate # Added default_collate
import hydra.utils
from omegaconf import DictConfig, OmegaConf
import hydra

# Use relative import as factory.py is in the same package level as base.py
from .base import BaseDataset 
# Remove direct import of specific dataset types unless absolutely needed for factory logic
# from .dataset import CharDataset # Example, adjust if others are needed
from .tokenizers.base import BaseTokenizer # Needed for collate setup
# Remove import for SentencePieceTokenizer if no longer needed directly here
# from .tokenizers.sentencepiece import SentencePieceTokenizer
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
                collate_fn=collate_fn,
                drop_last=True
            )
            self.dataloaders[split] = dataloader
            logger.info(f"Created DataLoader for split '{split}' with batch_size={batch_size}, shuffle={shuffle}, drop_last=True")
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

def prepare_dataloaders_from_config(
    config: DictConfig,
    tokenizer_override: Optional[BaseTokenizer] = None
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Optional[BaseTokenizer]]:
    """
    Prepares train, validation, and test DataLoaders along with the tokenizer
    based on a Hydra configuration object.
    Raises ValueError if instantiation fails for train or val splits or if required splits are missing.
    """
    train_loader, val_loader, test_loader = None, None, None
    tokenizer = tokenizer_override # Prioritize override

    if "data" not in config:
        logger.error("Configuration is missing the required 'data' section.")
        raise ValueError("Configuration is missing the required 'data' section.")
    data_cfg = config.data

    # Check for datasets section
    if "datasets" not in data_cfg:
         raise ValueError("Configuration missing 'data.datasets' section.")

    # Check required splits are defined if expected (assuming train/val required if present)
    required_splits = ["train", "val"]
    for req_split in required_splits:
        if req_split in data_cfg.datasets and data_cfg.datasets.get(req_split) is None:
             # This covers cases where the key exists but is null
             raise ValueError(f"Configuration for required split '{req_split}' is defined but null.")
        elif req_split in data_cfg.datasets and 'dataset' not in data_cfg.datasets[req_split]:
             # This covers cases where the split exists but lacks the 'dataset' subkey
             raise ValueError(f"Configuration for required split '{req_split}' is missing the 'dataset' key.")
        # Note: We don't raise if the *key* itself (e.g., 'train') is completely missing yet,
        # because the user might only configure 'test', for example.
        # The check later ensures the loader is created if the key exists.

    # 1. Instantiate Tokenizer
    if tokenizer is None and "tokenizer" in data_cfg:
        try:
            logger.info(f"Instantiating tokenizer from config: {data_cfg.tokenizer.get('_target_', 'N/A')}")
            tokenizer_cfg = data_cfg.tokenizer
            # Ensure it's a DictConfig for instantiation
            if not isinstance(tokenizer_cfg, DictConfig):
                 tokenizer_cfg = OmegaConf.create(tokenizer_cfg)
            
            # Standard instantiation - SentencePieceTokenizer __init__ now handles model_path
            tokenizer = hydra.utils.instantiate(tokenizer_cfg, _convert_="partial")
            
            if not isinstance(tokenizer, BaseTokenizer):
                 logger.warning(f"Instantiated tokenizer is not a BaseTokenizer subclass: {type(tokenizer)}. May cause issues.")
            logger.info(f"Successfully instantiated tokenizer: {type(tokenizer).__name__}")

        except Exception as e:
            logger.error(f"Failed to instantiate tokenizer from config: {e}", exc_info=True)
            tokenizer = None # Ensure tokenizer is None on failure
    elif tokenizer is not None:
         logger.info(f"Using provided tokenizer override: {type(tokenizer).__name__}")
    else:
         logger.info("No tokenizer override provided and no 'data.tokenizer' config found.")

    # 2. Instantiate Datasets and DataLoaders for each split
    for split in ["train", "val", "test"]:
        if split in data_cfg.datasets:
            logger.info(f"Processing '{split}' split...")
            split_cfg_full = data_cfg.datasets[split]

            # --- Ensure dataset config exists ---
            if 'dataset' not in split_cfg_full:
                # This should ideally be caught by the initial check for required splits,
                # but good to have a safeguard here too, especially for 'test'.
                logger.error(f"Missing 'dataset' key in config for split '{split}'. Skipping.")
                if split in required_splits:
                     raise ValueError(f"Configuration for required split '{split}' is missing the 'dataset' key.")
                continue # Skip optional splits like 'test' if misconfigured

            split_dataset_cfg_orig = split_cfg_full['dataset']

            # --- Instantiate Dataset ---
            logger.debug(f"Config for '{split}' dataset before instantiate: {OmegaConf.to_yaml(split_dataset_cfg_orig)}")
            try:
                # IMPORTANT: Create a copy and remove 'tokenizer' if it exists
                # This prevents passing it as an unexpected keyword arg to the dataset's __init__
                split_dataset_cfg_for_instantiate = OmegaConf.create(OmegaConf.to_container(split_dataset_cfg_orig, resolve=True)) # Create a resolved copy
                if 'tokenizer' in split_dataset_cfg_for_instantiate:
                    del split_dataset_cfg_for_instantiate['tokenizer']
                    logger.debug(f"Removed 'tokenizer' key before instantiating dataset for split '{split}'.")
                
                # Check if _target_ exists before attempting instantiation
                if '_target_' not in split_dataset_cfg_for_instantiate:
                    raise ValueError(f"Dataset configuration for split '{split}' is missing the '_target_' key.")

                # Instantiate Dataset using the modified config
                dataset = hydra.utils.instantiate(split_dataset_cfg_for_instantiate, _convert_="partial")

                # --- Validate Dataset Type ---
                if not isinstance(dataset, Dataset):
                     msg = f"Instantiated object for split '{split}' is not a PyTorch Dataset: {type(dataset)}"
                     logger.error(msg)
                     # Raise error immediately for required splits
                     if split in required_splits:
                         raise ValueError(msg)
                     continue # Skip optional splits like 'test' if instantiation yielded wrong type

                logger.info(f"Successfully instantiated dataset for split '{split}': {type(dataset).__name__}")
                if isinstance(dataset, BaseDataset):
                    dataset.summary()

                # --- Instantiate DataLoader ---
                logger.info(f"Instantiating dataloader for split '{split}'...")

                # Determine Collate Function (use default if none specified/found)
                collate_fn = getattr(dataset, 'collate_fn', None)
                if collate_fn and not callable(collate_fn):
                     logger.warning(f"Dataset for split '{split}' has a 'collate_fn' attribute, but it's not callable. Using default.")
                     collate_fn = default_collate # Use torch default
                elif collate_fn:
                     logger.info(f"Using collate_fn from dataset for split '{split}'.")
                else:
                     logger.info(f"No custom collate_fn found for split '{split}'. Using default torch collate.")
                     collate_fn = default_collate # Use torch default

                # Get dataloader configs, merging top-level and split-specific
                top_level_dataloader_cfg = data_cfg.get("dataloader", {})
                split_dataloader_cfg = split_cfg_full.get("dataloader", {})

                # Merge args, prioritizing split-specific config
                batch_size = split_dataloader_cfg.get("batch_size", top_level_dataloader_cfg.get("batch_size", 1)) # Default to 1 if unspecified
                num_workers = split_dataloader_cfg.get("num_workers", top_level_dataloader_cfg.get("num_workers", 0))
                pin_memory = split_dataloader_cfg.get("pin_memory", top_level_dataloader_cfg.get("pin_memory", torch.cuda.is_available()))
                drop_last = split_dataloader_cfg.get("drop_last", top_level_dataloader_cfg.get("drop_last", False))
                shuffle_default = (split == 'train') # Shuffle train by default
                shuffle = split_dataloader_cfg.get("shuffle", top_level_dataloader_cfg.get("shuffle", shuffle_default))

                loader_args = {
                    "batch_size": batch_size,
                    "num_workers": num_workers,
                    "pin_memory": pin_memory,
                    "drop_last": drop_last,
                    "shuffle": shuffle,
                    "collate_fn": collate_fn,
                    "dataset": dataset
                }
                
                logger.debug(f"DataLoader args for split '{split}': { {k:v for k,v in loader_args.items() if k != 'dataset'} }") # Don't log full dataset

                # Directly instantiate DataLoader
                loader = DataLoader(**loader_args)
                logger.info(f"Successfully instantiated dataloader for split '{split}'.")

                # Assign loader to the correct variable
                if split == "train":
                    train_loader = loader
                elif split == "val":
                    val_loader = loader
                elif split == "test":
                    test_loader = loader

            except Exception as e:
                 # Wrap and re-raise errors, especially for critical splits
                error_msg = f"Failed during dataset/dataloader instantiation for split '{split}': {e}"
                logger.error(error_msg, exc_info=True)
                # Raise immediately for required splits or if the error is critical
                if split in required_splits or isinstance(e, (ValueError, TypeError, hydra.errors.InstantiationException)):
                     # Re-raise crucial errors for required splits
                     raise ValueError(error_msg) from e
                # Potentially allow optional splits like 'test' to fail more gracefully
                # else:
                #    logger.warning(f"Optional split '{split}' failed, continuing...")

    # Final check: Ensure loaders exist IF their config was provided
    # This catches cases where instantiation failed silently for required splits (though previous checks should prevent this)
    if "train" in data_cfg.datasets and train_loader is None:
         raise RuntimeError("Train dataloader configuration exists but loader creation failed unexpectedly.")
    if "val" in data_cfg.datasets and val_loader is None:
         raise RuntimeError("Validation dataloader configuration exists but loader creation failed unexpectedly.")

    return train_loader, val_loader, test_loader, tokenizer


def create_dataset(config: Dict[str, Any]) -> BaseDataset:
    """
    Factory function to create a dataset instance based on configuration.

    Args:
        config: Dictionary or DictConfig containing dataset configuration, 
                including a `_target_` key.

    Returns:
        An instance of a BaseDataset subclass.

    Raises:
        ValueError: If `_target_` key is missing or instantiation fails.
    """
    logger.info(f"Attempting to create dataset from config: {config.get('_target_', 'N/A') if isinstance(config, (dict, DictConfig)) else 'Invalid Config Type'}")

    # Accept both dict and DictConfig
    if config is None or not isinstance(config, (dict, DictConfig)) or "_target_" not in config:
        raise ValueError(
            "Dataset configuration must be a dictionary or DictConfig with a '_target_' key. "
            f"Received type: {type(config)}"
        )
    
    # If it's a DictConfig, Hydra's instantiate utility prefers it directly.
    # If it's a dict, instantiate might still work, but passing DictConfig is safer.
    if isinstance(config, dict):
        config = OmegaConf.create(config) # Convert dict to DictConfig

    try:
        # Use Hydra utility to instantiate the class specified by _target_
        # _convert_="partial" allows deferred instantiation of nested configs if needed
        dataset_instance = hydra.utils.instantiate(config, _convert_="partial")
        
        # Validate the instantiated object is a PyTorch Dataset
        if not isinstance(dataset_instance, Dataset):
            raise TypeError(f"Instantiated object is not a PyTorch Dataset: {type(dataset_instance)}")
            
        logger.info(f"Successfully created dataset: {type(dataset_instance).__name__}")
        
        # Perform summary logging if it's our custom BaseDataset
        if isinstance(dataset_instance, BaseDataset):
            dataset_instance.summary() 
            
        return dataset_instance
    except hydra.errors.InstantiationException as e:
        logger.error(f"Hydra failed to instantiate dataset from config ({config.get('_target_', 'N/A')}): {e}", exc_info=False)
        logger.debug(f"Full stack trace for instantiation error:", exc_info=True)
        # Add more specific error messages based on common Hydra issues if possible
        if "missing value" in str(e).lower():
             logger.error("Instantiation failed likely due to a missing required argument in the config.")
        elif "cannot find class" in str(e).lower():
             logger.error(f"Instantiation failed because the target class '{config.get('_target_')}' could not be found. Check imports and paths.")
        raise ValueError(f"Could not create dataset from config: {e}") from e
    except TypeError as e:
         # Catch TypeErrors that might occur during instantiation (e.g., wrong arg types)
         logger.error(f"TypeError during dataset instantiation ({config.get('_target_', 'N/A')}): {e}", exc_info=True)
         raise ValueError(f"Could not create dataset due to a TypeError: {e}") from e
    except Exception as e: # Catch any other unexpected errors during instantiation
        logger.error(f"An unexpected error occurred during dataset instantiation ({config.get('_target_', 'N/A')}): {e}", exc_info=True)
        raise ValueError(f"Could not create dataset from config due to an unexpected error: {e}") from e


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
            split_cfg_full = data_cfg.datasets[split]
            
            # --- Extract the actual dataset config --- #
            if 'dataset' not in split_cfg_full:
                logger.error(f"Missing 'dataset' key in config for split '{split}'. Skipping.")
                continue
            split_dataset_cfg = split_cfg_full['dataset'] # Get the nested dataset config
            # --- End Extract --- #

            logger.debug(f"Config for '{split}' dataset before instantiate: {OmegaConf.to_yaml(split_dataset_cfg)}")
            try:
                # Instantiate Dataset using the extracted config
                dataset = hydra.utils.instantiate(split_dataset_cfg, _convert_="partial")
                
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