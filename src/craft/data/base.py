"""
Base dataset classes and abstractions for Craft.

This module defines base classes and utilities for dataset handling.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader
import hydra.utils
from omegaconf import DictConfig, OmegaConf
import hydra

# Import specific dataset types if needed for factory function fallbacks
# --- Remove top-level import to break cycle ---
# from src.data.dataset import CharDataset # Assuming CharDataset is in dataset.py

# Import the collate function and tokenizer
# from .collation import character_collate_fn
# from .tokenizer import CharacterTokenizer 
# from functools import partial # To create collate_fn with tokenizer

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
        # --- Remove validation from base class --- 
        # Let subclasses validate the specific keys they need.
        # self.data = None 
        # self._validate_config()
        
    # def _validate_config(self) -> None:
    #     """
    #     Validate the dataset configuration.
    #     
    #     Raises:
    #         ValueError: If required configuration options are missing
    #     """
    #     required_keys = ['paths', 'data'] # Example required keys
    #     missing_keys = [key for key in required_keys if key not in self.config]
    #     
    #     if missing_keys:
    #         raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
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
    
    # Optional placeholder for common preprocessing logic
    # Subclasses can override this if needed, or implement their own steps.
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
        # Default implementation does nothing.
        return sample

    # Potential future additions (kept abstract or optional for adaptability):
    # @abstractmethod
    # def load_data(self):
    #     """Loads the raw data from its source (e.g., file, database)."""
    #     pass
    
    # def get_tokenizer(self):
    #     """Returns a tokenizer associated with the dataset, if any."""
    #     return getattr(self, 'tokenizer', None)

    def summary(self):
        """Prints a basic summary of the dataset."""
        try:
            length = len(self)
            logging.info(f"Dataset: {self.__class__.__name__}, Length: {length}")
            # Potentially log more details from config if available
            if self.config:
                 logging.info(f"Config highlights: { {k: v for k, v in self.config.items() if k in ['data_path', 'split', 'tokenizer_name']} }") # Log only key configs
                 
        except NotImplementedError:
            logging.warning(f"Dataset: {self.__class__.__name__} - __len__ not implemented yet.")
        except Exception as e:
            logging.error(f"Error generating summary for {self.__class__.__name__}: {e}")

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


def create_dataset_from_config(data_config: DictConfig, split_config: DictConfig, original_cwd: Optional[str] = None, split_name: str = "unknown") -> BaseDataset:
    """
    Factory function to create a dataset instance from configuration dictionaries.
    Resolves relative paths using original_cwd if provided.

    Args:
        data_config (DictConfig): The main data configuration (e.g., cfg.data).
        split_config (DictConfig): Configuration for the specific split dataset (e.g., cfg.data.train.dataset).
        original_cwd (Optional[str]): The original working directory to resolve relative paths.
        split_name (str): The name of the split being processed (e.g., 'train').

    Returns:
        BaseDataset: An instance of the dataset defined by the configuration.

    Raises:
        ValueError: If instantiation fails or required configuration is missing.
    """
    if not split_config or '_target_' not in split_config:
        raise ValueError("Dataset configuration must contain a '_target_' key.")

    # Determine the target class path
    target_class_path = split_config._target_
    if not target_class_path:
        # Fallback: Infer based on data_cfg or default?
        # For now, require _target_ in the split config for clarity.
        raise ValueError(f"Dataset target class ('_target_') not specified in configuration for split '{split_name}'.")

    # --- Special Handling (Example for CharDataset - REMOVE LATER) ---
    # if target_class_path == "src.data.dataset.CharDataset":
    #     # CharDataset might need specific args like vocab_path resolved
    #     if 'vocab_path' in split_cfg and not os.path.isabs(split_cfg.vocab_path):
    #         split_cfg.vocab_path = os.path.join(original_cwd, split_cfg.vocab_path)
    #         logger.debug(f"Resolved relative vocab_path for CharDataset: {split_cfg.vocab_path}")
    # --- End Special Handling ---

    logger.info(f"Attempting to create dataset for split using Hydra target: {target_class_path}...")

    # Resolve relative file path within the split config
    # Use resolve=True to interpolate any OmegaConf variables first
    instant_config = OmegaConf.to_container(split_config, resolve=True)

    try:
        # Merge necessary top-level keys if not present
        if target_class_path == "src.data.dataset.CharDataset": # TODO: CharDataset should be removed should be removed. Why is this stil here?
            for key in ['vocab_path', 'block_size']:
                 if key not in instant_config and key in data_config:
                     instant_config[key] = data_config.get(key)
                     logger.info(f"Added {key} from data_config: {instant_config[key]}")

        # --- Resolve Paths using original_cwd ---
        if original_cwd:
            logger.debug(f"Attempting to resolve paths relative to CWD: {original_cwd}")
            for key in ['file_path', 'vocab_path']:
                # Check if key exists and its value can be treated as a path
                if key in instant_config and instant_config[key] is not None:
                    potential_path = str(instant_config[key]) # Convert to string just in case
                    if potential_path: # Ensure it's not an empty string
                        if not os.path.isabs(potential_path):
                            # Construct absolute path relative to the original CWD
                            absolute_path = os.path.join(original_cwd, potential_path)
                            # Check if the resolved path actually exists
                            if os.path.exists(absolute_path):
                                logger.info(f"Resolving relative path for '{key}': '{potential_path}' -> '{absolute_path}'")
                                instant_config[key] = absolute_path # Update the dict with the absolute path
                            else:
                                # Log a warning if the resolved path doesn't exist, but keep the original relative path
                                logger.warning(f"Resolved path for '{key}' ('{absolute_path}') does not exist. Keeping original path: '{potential_path}'. Check config if this is unexpected.")
                        else:
                            logger.debug(f"Path for '{key}' ('{potential_path}') is already absolute. No change needed.")
                    else:
                        logger.debug(f"Key '{key}' value is empty after string conversion. Skipping path resolution.")
                elif key in instant_config:
                     logger.debug(f"Key '{key}' found but value is None. Skipping path resolution.")
                # else: key not present, nothing to resolve
        else:
             logger.warning("original_cwd not provided to create_dataset_from_config. Cannot resolve relative paths.")
        # --- End Resolve Paths ---

        # Convert the dataset_config subset to OmegaConf DictConfig for instantiation
        if not isinstance(instant_config, DictConfig):
            instant_config = OmegaConf.create(instant_config)

        # Original logging line using attribute access, assuming instant_config is now DictConfig
        logger.info(f"Instantiating {instant_config._target_} with config: {OmegaConf.to_container(instant_config)}")
        # Use the OmegaConf object for instantiation
        dataset = hydra.utils.instantiate(instant_config)

        logger.info(f"Dataset '{split_name}' instantiated successfully: {type(dataset)}")
        return dataset
    except Exception as e:
        # Log the config *before* conversion back to OmegaConf for better debugging dict structure
        logger.error(f"Failed to instantiate dataset from config (dict form used for path resolution): {instant_config}. Error: {e}", exc_info=True)
        raise


def prepare_dataloaders_from_config(data_config: DictConfig, batch_size: int, num_workers: int = 0, original_cwd: Optional[str] = None) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Prepares train, validation, and test dataloaders based on the data configuration.
    (Reverted signature to return only dataloaders)

    Args:
        data_config (DictConfig): The data configuration section (e.g., cfg.data).
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): Number of worker processes for loading data.
        original_cwd (Optional[str]): The original working directory before Hydra.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]: 
            Train, validation, and test dataloaders. Returns None for splits not defined in the config.
    """
    logger.info("--- Entering prepare_dataloaders_from_config ---")
    train_loader, val_loader, test_loader = None, None, None
    # train_dataset = None # REMOVED - No longer need to extract vocab_size here
    # vocab_size = None    # REMOVED

    for split in ["train", "val", "test"]:
        # Check if the attribute exists on the Pydantic model and is not None
        if hasattr(data_config, split) and getattr(data_config, split) is not None:
            logger.info(f"Preparing dataset and dataloader for split: {split}")
            # Get the config for the split using getattr
            split_config = getattr(data_config, split)
            dataset = None # Initialize dataset to None for this split
            try:
                # Pass top-level data_config and split_config to factory
                # Note: split_config here is DatasetSplitConfig, we need its 'dataset' part
                if hasattr(split_config, 'dataset') and split_config.dataset is not None:
                    dataset_specific_config = split_config.dataset 
                    # Ensure dataset_specific_config is a DictConfig if needed by create_dataset_from_config
                    if isinstance(dataset_specific_config, dict):
                         dataset_specific_config = OmegaConf.create(dataset_specific_config)
                    elif not isinstance(dataset_specific_config, DictConfig):
                         logger.error(f"Dataset config for split '{split}' has unexpected type: {type(dataset_specific_config)}")
                         raise ValueError(f"Invalid dataset config type for split '{split}'.")
                    
                    # Pass the split name to the factory function
                    dataset = create_dataset_from_config(
                        data_config=data_config, 
                        split_config=dataset_specific_config, 
                        original_cwd=original_cwd,
                        split_name=split # Pass the current split name
                    )
                else:
                    logger.error(f"Split config for '{split}' is missing the 'dataset' attribute or it is None.")
                    raise ValueError(f"Invalid split config for '{split}'.")

            except Exception as e: # Catch any exception during dataset creation
                logger.error(f"Failed to create dataset for split '{split}'. See underlying error: {e}", exc_info=True)
                if split == "train": # Failure to create train dataset is critical
                    logger.error("Cannot proceed without the training dataset. Returning None dataloaders.")
                    return None, None, None # Return None immediately if train fails
                else: # For val/test, log and skip this split
                    logger.warning(f"Skipping dataloader creation for optional split '{split}' due to dataset error.")
                    continue # Go to the next split
            
            # Check if dataset was created successfully and is not empty
            if dataset is None or len(dataset) == 0:
                logger.warning(f"Dataset for split '{split}' is None or empty. Skipping DataLoader creation.")
                if split == "train": # Also critical if train dataset is empty
                     logger.error("Training dataset is empty. Returning None dataloaders.")
                     return None, None, None
                else:
                    continue

            shuffle = (split == "train")
            logger.info(f"DataLoader shuffle set to {shuffle} for split '{split}'")
            
            # Revert DataLoader creation to use default collate_fn
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=None # Use default collate
            )
            
            if split == "train":
                train_loader = loader
            elif split == "val":
                val_loader = loader
            elif split == "test":
                test_loader = loader

    # REMOVED: Logic to extract and log vocab_size
    # if train_dataset is not None:
    #     if hasattr(train_dataset, 'vocab_size'):
    #         ...
    #     else:
    #         ...
    # else:
    #     ...
        
    logger.info(f"--- Exiting prepare_dataloaders_from_config. Returning: train={train_loader is not None}, val={val_loader is not None}, test={test_loader is not None} ---")
    return train_loader, val_loader, test_loader # Return only the 3 dataloaders


def create_data_manager_from_config(config: DictConfig) -> DataManager:
    # ... (create_data_manager_from_config implementation remains the same) ...
    pass # Keep the existing function implementation 


def create_data_loaders_from_config(data_config: Union[DictConfig, dict]) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for train, validation, and optionally test splits based
    on a provided configuration dictionary or DictConfig.

    Args:
        data_config: The configuration dictionary or DictConfig object for the 'data' section.
                     Each split config should have a `dataset` sub-config for instantiation
                     and an optional `dataloader` sub-config for loader parameters.

    Returns:
        A dictionary mapping split names ('train', 'val', 'test') to their
        corresponding DataLoader instances.

    Raises:
        ValueError: If required configurations (like train/val splits or _target_)
                    are missing or invalid.
        hydra.errors.InstantiationException: If Hydra fails to instantiate the dataset.
    """
    logger.info("Creating DataLoaders...")
    dataloaders = {}

    # --- Check for required train and val splits --- 
    if "train" not in data_config:
        raise ValueError("'train' split configuration is required in the 'data' section.")
    if "val" not in data_config:
        raise ValueError("'val' split configuration is required in the 'data' section.")

    # --- Default DataLoader parameters (can be overridden per split) ---
    default_batch_size = data_config.get("batch_size", 32)
    default_num_workers = data_config.get("num_workers", 0)
    default_pin_memory = data_config.get("pin_memory", True)

    for split in ["train", "val", "test"]:
        logger.info(f"Processing '{split}' split...")
        split_cfg = data_config.get(split)

        if split_cfg is None:
            # Only skip if it's the optional 'test' split
            if split == "test":
                logger.warning(f"No configuration found for optional split '{split}'. Skipping dataloader creation.")
                continue
            else:
                # This case should be caught by the checks above, but added for safety
                raise ValueError(f"Configuration for required split '{split}' is missing.")

        if not isinstance(split_cfg, (DictConfig, dict)): # Allow both DictConfig and standard dict
            logger.error(f"Configuration for split '{split}' must be a dictionary/DictConfig. Found type: {type(split_cfg)}")
            raise ValueError(f"Invalid configuration type for split '{split}'.")

        # --- Get Dataset Config ---
        dataset_cfg = split_cfg.get("dataset")
        if dataset_cfg is None or not isinstance(dataset_cfg, (DictConfig, dict)): # Allow dict
             raise ValueError(f"Missing or invalid 'dataset' configuration section for split '{split}'.")

        # Check for _target_ in a way that works for both DictConfig and dict
        target_key = "_target_" # Hydra's key
        if isinstance(dataset_cfg, dict) and target_key not in dataset_cfg:
             raise ValueError(f"Missing '{target_key}' key in 'dataset' configuration dictionary for split '{split}'.")
        # Optional: Keep the elif check for DictConfig if desired
        elif isinstance(dataset_cfg, DictConfig) and target_key not in dataset_cfg:
             # This case might be less likely now but kept for safety
             raise ValueError(f"Missing '{target_key}' key in 'dataset' DictConfig for split '{split}'.")

        # --- Get DataLoader Config --- 
        dataloader_cfg = split_cfg.get("dataloader", {}) # Default to empty dict
        if not isinstance(dataloader_cfg, DictConfig):
            logger.warning(f"'dataloader' config for split '{split}' is not a DictConfig. Using defaults. Found: {type(dataloader_cfg)}")
            # Ensure dataloader_cfg is a dict if we proceed
            if not isinstance(dataloader_cfg, dict):
                 dataloader_cfg = {}

        try:
            # Use dictionary access for target, as dataset_cfg might be dict
            target_path = dataset_cfg.get("_target_", "[N/A]") # Use .get() for safety
            logger.info(f"Instantiating dataset for split '{split}' with target: {target_path}")
            # Instantiate using the dataset sub-config
            # hydra.utils.instantiate works fine with dicts too
            dataset: Dataset = hydra.utils.instantiate(dataset_cfg)
            logger.info(f"Dataset '{split}' instantiated successfully: {type(dataset)}")

            # ---> Add assertion here <---
            assert len(dataset) > 0, f"Dataset length for split '{split}' is not positive: {len(dataset)}"

            # Determine DataLoader parameters, prioritizing split-specific config
            batch_size = dataloader_cfg.get("batch_size", default_batch_size)
            num_workers = dataloader_cfg.get("num_workers", default_num_workers)
            pin_memory = dataloader_cfg.get("pin_memory", default_pin_memory)

            # Default shuffle logic (True for train, False otherwise), can be overridden
            default_shuffle = (split == "train")
            shuffle = dataloader_cfg.get("shuffle", default_shuffle)

            # Default drop_last logic (True for train, False otherwise), can be overridden
            default_drop_last = (split == "train")
            drop_last = dataloader_cfg.get("drop_last", default_drop_last)

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last
            )
            dataloaders[split] = loader
            logger.info(f"DataLoader for split '{split}' created. Shuffle={shuffle}, DropLast={drop_last}, BatchSize={batch_size}, NumWorkers={num_workers}")

        except hydra.errors.InstantiationException as e:
            # Ensure dataset_cfg is convertible for logging if it's a dict
            loggable_dataset_cfg = OmegaConf.create(dataset_cfg) if isinstance(dataset_cfg, dict) else dataset_cfg
            logger.error(f"Hydra failed to instantiate dataset for split '{split}'. Config: {OmegaConf.to_yaml(loggable_dataset_cfg, resolve=True)}. Error: {e}", exc_info=True)
            # Re-raise the specific Hydra exception for clarity
            raise e
        except Exception as e:
            # Ensure dataset_cfg is convertible for logging if it's a dict
            loggable_dataset_cfg = OmegaConf.create(dataset_cfg) if isinstance(dataset_cfg, dict) else dataset_cfg
            logger.error(f"Failed to create DataLoader for split '{split}'. Dataset Config: {OmegaConf.to_yaml(loggable_dataset_cfg, resolve=True)}. Error: {e}", exc_info=True)
            # Raise a more general error if it wasn't an instantiation issue
            raise ValueError(f"Failed to create loader for split '{split}'") from e

    logger.info("DataLoaders creation finished.")
    if not dataloaders:
        # This state should ideally not be reached if train/val are required
        logger.warning("No DataLoaders were created despite checks. Review logic.")

    return dataloaders


# Ensure this function is imported where needed, potentially in src/craft/data/__init__.py