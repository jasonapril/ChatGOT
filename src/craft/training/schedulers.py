"""
Factory function for creating learning rate schedulers.
"""
import logging
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
from omegaconf import DictConfig

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
        logger.info("No scheduler configuration provided, skipping scheduler creation.")
        return None

    scheduler_name = scheduler_cfg.get("_target_")
    if not scheduler_name:
        raise ValueError("Scheduler configuration must specify '_target_' (e.g., 'torch.optim.lr_scheduler.CosineAnnealingLR')")

    # Filter out the target key, pass the rest as kwargs
    scheduler_kwargs = {k: v for k, v in scheduler_cfg.items() if k != "_target_"}

    logger.info(f"Creating scheduler: {scheduler_name} with params: {scheduler_kwargs}")

    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    try:
        # --- Explicitly supported schedulers ---
        if scheduler_name.lower() == "torch.optim.lr_scheduler.cosineannealinglr" or scheduler_name.lower() == "cosineannealinglr":
            if 'T_max' not in scheduler_kwargs:
                raise ValueError("CosineAnnealingLR configuration must include 'T_max'")
            # Explicitly check type before instantiation
            t_max = scheduler_kwargs['T_max']
            if not isinstance(t_max, int):
                 raise ValueError(f"CosineAnnealingLR parameter 'T_max' must be an integer, got {type(t_max)}")
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        
        # --- Add other explicitly supported schedulers here with elif --- 
        # elif scheduler_name.lower() == "torch.optim.lr_scheduler.steplr":
            # ... validation and instantiation ...
            
        else:
            # --- Unsupported scheduler type ---
            raise ValueError(f"Unsupported scheduler type: {scheduler_name}. Add explicit support or check config.")

    except ValueError as ve:
        # Catch explicit ValueErrors raised above (missing required args, unsupported type, wrong type)
        logger.error(f"Configuration error for scheduler {scheduler_name}: {ve}")
        raise # Re-raise the specific configuration error
        
    except TypeError as te:
        # Catch unexpected TypeError from underlying scheduler instantiation (should be less likely now)
        logger.error(f"Unexpected TypeError during scheduler instantiation for {scheduler_name}: {te}. Check config parameters.")
        raise ValueError(f"Invalid parameters for scheduler {scheduler_name}: {te}") from te
        
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"Failed to create scheduler {scheduler_name} due to unexpected error: {e}")
        raise ValueError(f"Unexpected error creating scheduler {scheduler_name}") from e

    # Should only reach here if scheduler is successfully created
    if scheduler is None: 
        # This case should ideally not happen if logic above is correct, but as a safeguard:
        raise ValueError(f"Scheduler creation failed unexpectedly for {scheduler_name}")
        
    logger.info(f"Scheduler {type(scheduler).__name__} created successfully.")
    return scheduler 