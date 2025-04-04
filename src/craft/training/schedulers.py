"""
Factory function for creating learning rate schedulers.
"""
import logging
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)

def create_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Optional[DictConfig]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Creates a learning rate scheduler based on the provided configuration.

    Args:
        optimizer: The optimizer instance whose learning rate will be scheduled.
        scheduler_cfg: The OmegaConf DictConfig object for the scheduler, 
                       or None if no scheduler is configured. If provided, 
                       expected to have '_target_' and relevant parameters.

    Returns:
        An instantiated PyTorch LR scheduler, or None if scheduler_cfg is None.

    Raises:
        ValueError: If the scheduler type specified by '_target_' is unknown 
                    or if required parameters are missing.
    """
    if scheduler_cfg is None:
        logger.info("No scheduler configuration provided. Skipping scheduler creation.")
        return None

    # Ensure target key is present (check both 'target' and '_target_')
    target_key_alias = "target"
    target_key_hydra = "_target_"
    target_path = scheduler_cfg.get(target_key_alias) or scheduler_cfg.get(target_key_hydra)

    if not target_path:
        raise ValueError(f"Scheduler configuration must specify '{target_key_alias}' or '{target_key_hydra}'.")

    # Filter out the target key(s), pass the rest as kwargs
    scheduler_kwargs = scheduler_cfg.copy() # Make a copy
    scheduler_kwargs.pop(target_key_alias, None)
    scheduler_kwargs.pop(target_key_hydra, None)

    logger.info(f"Creating scheduler: {target_path} with params: {scheduler_kwargs}")

    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    try:
        # Check for supported schedulers (case-insensitive and allow short names)
        target_lower = target_path.lower()
        if "cosineannealinglr" in target_lower:
            # Ensure required parameters are present (e.g., T_max)
            if 'T_max' not in scheduler_kwargs:
                raise ValueError("CosineAnnealingLR config must include 'T_max'")
            # Optional: Validate T_max type
            if not isinstance(scheduler_kwargs['T_max'], int):
                raise ValueError("CosineAnnealingLR parameter 'T_max' must be an integer")
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
            logger.info("Scheduler CosineAnnealingLR created successfully.")

        else:
            logger.warning(f"Scheduler '{target_path}' not explicitly handled. Add support if needed.")
            raise ValueError(f"Unsupported or unrecognized scheduler type: {target_path}")

    except ValueError as ve:
        # Log the specific parameter error and re-raise
        logger.error(f"Failed to create scheduler {target_path} due to config error: {ve}")
        raise ve 
    except TypeError as te:
        # Catch potential TypeError from PyTorch if kwargs are wrong type
        logger.error(f"Failed to create scheduler {target_path} due to incorrect parameter type: {te}")
        raise TypeError(f"Incorrect parameter type for {target_path}: {te}") from te
    except Exception as e:
        logger.error(f"An unexpected error occurred creating scheduler {target_path}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error creating scheduler {target_path}") from e

    return scheduler

    # Should only reach here if scheduler is successfully created
    if scheduler is None: 
        # This case should ideally not happen if logic above is correct, but as a safeguard:
        raise ValueError(f"Scheduler creation failed unexpectedly for {target_path}")
        
    logger.info(f"Scheduler {type(scheduler).__name__} created successfully.")
    return scheduler 