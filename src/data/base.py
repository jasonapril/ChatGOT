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
from omegaconf import DictConfig
from omegaconf import OmegaConf

# Import specific dataset types if needed for factory function fallbacks
# --- Remove top-level import to break cycle ---
# from src.data.dataset import CharDataset # Assuming CharDataset is in dataset.py

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


def create_dataset_from_config(config: DictConfig, split: str) -> BaseDataset:
    """
    Factory function to create a dataset instance from a configuration.
    Prioritizes Hydra instantiation using `_target_` if present.
    Includes fallback logic for known simple types like CharDataset.

    Args:
        config (DictConfig): The dataset configuration (potentially nested under a split).
        split (str): The dataset split (e.g., 'train', 'val', 'test').

    Returns:
        BaseDataset: An instance of a dataset.
        
    Raises:
        ValueError: If the dataset cannot be instantiated.
    """
    logger.info(f"Attempting to create dataset for split '{split}'...")
    
    # Check for Hydra instantiation target first
    if "_target_" in config:
        logger.info(f"Instantiating dataset using Hydra target: {config._target_}")
        try:
            # Remove split=split argument as simplified CharDataset constructor doesn't take it
            dataset = hydra.utils.instantiate(config)
            if not isinstance(dataset, BaseDataset):
                logger.warning(f"Instantiated object of type {type(dataset)} is not a BaseDataset subclass.")
            return dataset
        except Exception as e:
            logger.error(f"Hydra instantiation failed for target {config._target_}: {e}", exc_info=True)
            raise ValueError(f"Could not instantiate dataset from config using _target_: {config._target_}") from e

    # Fallback logic for known types if _target_ is not specified
    logger.warning(f"No '_target_' found in dataset config for split '{split}'. Attempting fallback instantiation.")
    data_format = config.get("format", "unknown") # Use .get for safer access

    # Example fallback for CharDataset (adjust condition as needed)
    if data_format == "character":
        # --- Import locally to break circular dependency ---
        from src.data.dataset import CharDataset 
        logger.info(f"Using fallback logic to create CharDataset.")
        try:
            # Extract specific args needed by CharDataset using attribute access
            file_path = config.file_path  # Use attribute access
            block_size = config.block_size # Use attribute access
            
            # Validate required args for fallback path
            if file_path is None or block_size is None:
                # This check might be less necessary with attribute access if keys are guaranteed
                raise ValueError("Fallback for CharDataset requires 'file_path' and 'block_size' in config.")
                
            # Pass other config items as kwargs
            other_kwargs = {k: v for k, v in config.items() if k not in ['file_path', 'block_size', 'format', '_target_']} # Exclude _target_ too

            # Call with specific args + remaining config as kwargs
            dataset = CharDataset(file_path=file_path, block_size=block_size, **other_kwargs) 
            return dataset
        except Exception as e:
            logger.error(f"Failed to instantiate CharDataset using fallback: {e}", exc_info=True)
            # Propagate the original error if it's specific, or the generic one
            if isinstance(e, (ValueError, TypeError)): 
                 raise # Reraise validation/type errors from CharDataset init
            raise ValueError("Fallback instantiation for CharDataset failed.") from e
    
    # Add other fallbacks here if necessary
    # elif data_format == "some_other_format":
    #     logger.info("Using fallback for SomeOtherDataset...")
    #     # dataset = SomeOtherDataset(...) 
    #     # return dataset
    
    logger.error(f"Could not create dataset for split '{split}'. No '_target_' specified and no matching fallback logic for format '{data_format}'.")
    raise ValueError(f"Unsupported dataset configuration for split '{split}': format '{data_format}'")


def prepare_dataloaders_from_config(data_config: DictConfig, batch_size: int, num_workers: int = 0, original_cwd: Optional[str] = None) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Prepares train, validation, and test dataloaders based on the data configuration.

    Args:
        data_config (DictConfig): The data configuration section (e.g., cfg.data).
        batch_size (int): The batch size for the dataloaders.
        num_workers (int): Number of worker processes for loading data.
        original_cwd (Optional[str]): The original working directory before Hydra.

    Returns:
        Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]: 
            Train, validation, and test dataloaders. Returns None for splits not defined in the config.
    """
    train_loader, val_loader, test_loader = None, None, None
    
    # --- Removed Tokenizer Loading --- 
    # Tokenizer loading/handling will be managed elsewhere (e.g., in Trainer or model)
    # tokenizer = None # Placeholder
    # if "tokenizer" in data_config and data_config.tokenizer.get("_target_"):
    #     logger.info(f"Loading tokenizer: {data_config.tokenizer._target_}")
    #     try:
    #         tokenizer = hydra.utils.instantiate(data_config.tokenizer)
    #     except Exception as e:
    #         logger.error(f"Failed to instantiate tokenizer: {e}", exc_info=True)
    #         # Decide whether to raise or continue without tokenizer
    # else:
    #     logger.warning("No tokenizer configuration found or _target_ missing in data_config.tokenizer")
        
    for split in ["train", "val", "test"]:
        if split in data_config:
            logger.info(f"Preparing dataset and dataloader for split: {split}")
            split_config = data_config[split]
            
            # --- Construct Absolute Path --- 
            # If original_cwd is provided and file_path is relative, construct absolute path
            if original_cwd and "file_path" in split_config and not os.path.isabs(split_config.file_path):
                relative_path = split_config.file_path
                absolute_path = os.path.join(original_cwd, relative_path)
                logger.info(f"Resolving relative path '{relative_path}' to '{absolute_path}'")
                # Create a mutable copy to modify
                mutable_split_config = OmegaConf.to_container(split_config, resolve=True)
                mutable_split_config['file_path'] = absolute_path
                # Convert back to DictConfig if needed by downstream, or pass the dict
                split_config = OmegaConf.create(mutable_split_config)
            # -------------------------------
            
            try:
                # Pass only the config, remove tokenizer argument
                dataset = create_dataset_from_config(config=split_config, split=split)
            except ValueError as e:
                logger.error(f"Failed to create dataset for split '{split}': {e}")
                # Depending on strictness, either raise e or continue
                continue # Skip this split if dataset creation fails
            
            if len(dataset) == 0:
                logger.warning(f"Dataset for split '{split}' is empty. Skipping DataLoader creation.")
                continue
                
            # Determine shuffle based on split
            shuffle = (split == "train")
            logger.info(f"DataLoader shuffle set to {shuffle} for split '{split}'")
            
            # Use default collate_fn for now. 
            # torch.utils.data.default_collate handles dicts of tensors correctly
            # by stacking tensors for each key. This works for fixed-size sequence
            # datasets like the current CharDataset.
            # TODO: Implement flexible collate function selection (e.g., via config)
            #       for handling padding (variable lengths) or on-the-fly tokenization.
            collate_fn = None 
            
            # --- Tokenizer Strategy --- 
            # For datasets requiring external tokenizers (e.g., Hugging Face BPE):
            # 1. Configuration: Define a tokenizer config section (e.g., cfg.tokenizer) 
            #    with `_target_` pointing to the tokenizer class (e.g., transformers.AutoTokenizer.from_pretrained)
            #    and necessary args (e.g., `pretrained_model_name_or_path`).
            # 2. Loading: Instantiate the tokenizer in the main training script: `tokenizer = hydra.utils.instantiate(cfg.tokenizer)`.
            # 3. Application: Pass the loaded `tokenizer` instance to a custom `collate_fn`.
            # 4. Collate Function: The custom `collate_fn` would:
            #    - Receive a list of samples (e.g., dictionaries with raw text) from the DataLoader.
            #    - Use the `tokenizer` to convert text to input_ids, add special tokens, etc.
            #    - Pad sequences to a consistent length within the batch (e.g., using tokenizer.pad).
            #    - Return the batch as a dictionary of tensors (e.g., {'input_ids': ..., 'attention_mask': ...}).
            # 5. DataLoader: Pass the custom `collate_fn` here instead of `None`.
            # --------------------------
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(), # Pin memory if using GPU
                collate_fn=collate_fn 
            )
            
            if split == "train":
                train_loader = loader
            elif split == "val":
                val_loader = loader
            elif split == "test":
                test_loader = loader
        else:
            logger.info(f"Split '{split}' not defined in data configuration.")
            
    return train_loader, val_loader, test_loader


def create_data_manager_from_config(config: DictConfig) -> DataManager:
    # ... (create_data_manager_from_config implementation remains the same) ...
    pass # Keep the existing function implementation 