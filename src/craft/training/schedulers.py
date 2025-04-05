"""
Factory function for creating learning rate schedulers.
"""
import logging
import torch.optim as optim
from typing import Optional, Union, Dict, Any
from omegaconf import DictConfig, OmegaConf
import hydra.utils

# Import the Pydantic config model
from ..config.schemas import SchedulerConfig

logger = logging.getLogger(__name__)

def create_scheduler(optimizer: optim.Optimizer, scheduler_cfg: Optional[SchedulerConfig]) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Creates a learning rate scheduler based on the provided Pydantic configuration.

    Args:
        optimizer: The optimizer instance whose learning rate will be scheduled.
        scheduler_cfg: An Optional SchedulerConfig Pydantic model instance containing
                       configuration like target, T_max, eta_min, etc.

    Returns:
        An instantiated PyTorch LR scheduler, or None if scheduler_cfg is None.

    Raises:
        ValueError: If the scheduler type specified by 'target' is unknown
                    or if required parameters are missing.
        AttributeError: If expected fields are missing in scheduler_cfg.
    """
    if scheduler_cfg is None:
        logger.info("No scheduler configuration provided. Skipping scheduler creation.")
        return None

    # Access target directly from the Pydantic model
    try:
        target_path = scheduler_cfg.target
    except AttributeError:
        raise ValueError("Scheduler configuration (SchedulerConfig) must have a 'target' field.")

    # --- Get Scheduler Keyword Arguments --- #
    # Convert Pydantic model to dict, excluding 'target'
    scheduler_kwargs = scheduler_cfg.model_dump(exclude={'target'})

    logger.info(f"Creating scheduler: {target_path} with params: {scheduler_kwargs}")

    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    try:
        # Get the scheduler class using Hydra's utility
        scheduler_cls = hydra.utils.get_class(target_path)

        # Instantiate the scheduler class
        # PyTorch schedulers take the optimizer as the first argument
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

    except ImportError:
        logger.error(f"Could not import scheduler class: {target_path}")
        raise ValueError(f"Unsupported scheduler target: {target_path}") from ImportError
    except TypeError as e:
        # Catch errors like missing required args or wrong types during instantiation
        logger.error(f"TypeError during scheduler instantiation for {target_path}: {e}. Check config: {scheduler_kwargs}")
        raise ValueError(f"Invalid parameters for scheduler {target_path}: {e}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred creating scheduler {target_path}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error creating scheduler {target_path}") from e

    # Should only reach here if scheduler is successfully created
    if scheduler is None:
        # This case should ideally not happen if logic above is correct, but as a safeguard:
        raise ValueError(f"Scheduler creation failed unexpectedly for {target_path}")

    logger.info(f"Scheduler {type(scheduler).__name__} created successfully.")
    return scheduler 