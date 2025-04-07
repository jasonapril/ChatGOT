# src/craft/data/utils.py
import logging
from typing import Dict, Any, Optional, Callable
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

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
        else:
             logger.info("Using default torch collate.")


        dataloader_instance = DataLoader(**loader_args)
        logger.info(f"Successfully created DataLoader with batch size {loader_args['batch_size']}")
        return dataloader_instance
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}", exc_info=True)
        raise ValueError(f"Could not create DataLoader: {e}") from e 