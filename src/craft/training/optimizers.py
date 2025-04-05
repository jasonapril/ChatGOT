"""
Factory function for creating optimizers.
"""
import logging
import torch
import torch.optim as optim
from typing import Dict, Any, Optional, Union
from omegaconf import DictConfig, OmegaConf
import hydra.utils

# Import the Pydantic config model
from ..config.schemas import OptimizerConfig

logger = logging.getLogger(__name__)

def create_optimizer(model: torch.nn.Module, optimizer_cfg: OptimizerConfig) -> optim.Optimizer:
    """
    Creates a PyTorch optimizer based on the provided Pydantic configuration.

    Args:
        model: The model whose parameters need optimization.
        optimizer_cfg: An OptimizerConfig Pydantic model instance containing configuration
                       like target, lr, weight_decay, etc.

    Returns:
        An instantiated PyTorch optimizer.

    Raises:
        ValueError: If the optimizer type specified by 'target' is unknown
                    or if required parameters are missing.
        AttributeError: If expected fields are missing in optimizer_cfg.
    """
    # Access target directly from the Pydantic model
    # Pydantic automatically handles the _target_ alias via validation_alias='_target_'
    try:
        target_path = optimizer_cfg.target
    except AttributeError:
        raise ValueError("Optimizer configuration (OptimizerConfig) must have a 'target' field.")

    # --- Get Optimizer Keyword Arguments --- #
    # Convert Pydantic model to dict, excluding the 'target' field
    # Use model_dump to get a dict, respecting aliases if needed
    # We exclude 'target' as it's not an optimizer parameter itself
    optimizer_kwargs = optimizer_cfg.model_dump(exclude={'target'})
    
    # --- DEBUG --- 
    # logger.debug(f"[DEBUG][Optimizer] Target Path: {target_path}")
    # logger.debug(f"[DEBUG][Optimizer] Kwargs: {optimizer_kwargs}")
    # --- END DEBUG ---

    logger.info(f"Creating optimizer: {target_path} with params: {optimizer_kwargs}")

    try:
        # Get the optimizer class using Hydra's utility
        optimizer_cls = hydra.utils.get_class(target_path)

        # Filter model parameters that require gradients
        params_to_optimize = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Optimizing {len(params_to_optimize)} parameter groups.")

        # Instantiate the optimizer class
        optimizer = optimizer_cls(params_to_optimize, **optimizer_kwargs)

    except ImportError:
        logger.error(f"Could not import optimizer class: {target_path}")
        raise ValueError(f"Unsupported optimizer target: {target_path}") from ImportError
    except TypeError as e:
        # Catch errors like missing required args or wrong types during instantiation
        logger.error(f"TypeError during optimizer instantiation for {target_path}: {e}. Check config: {optimizer_kwargs}")
        raise ValueError(f"Invalid parameters for optimizer {target_path}: {e}") from e
    except Exception as e:
        logger.error(f"Failed to create optimizer {target_path}: {e}", exc_info=True)
        raise

    logger.info(f"Optimizer {type(optimizer).__name__} created successfully.")
    return optimizer 