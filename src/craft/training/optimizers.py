"""
Factory function for creating optimizers.
"""
import logging
from typing import Dict, Any, Iterable

import torch
import torch.optim as optim
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def create_optimizer(model: torch.nn.Module, optim_cfg: DictConfig) -> optim.Optimizer:
    """
    Creates an optimizer based on the provided configuration.

    Args:
        model: The model whose parameters need optimization.
        optim_cfg: The OmegaConf DictConfig object for the optimizer, 
                   expected to have '_target_' and parameters like 'lr'.

    Returns:
        An instantiated PyTorch optimizer.

    Raises:
        ValueError: If the optimizer type specified by '_target_' is unknown 
                    or if required parameters are missing.
    """
    optimizer_name = optim_cfg.get("_target_")
    if not optimizer_name:
        raise ValueError("Optimizer configuration must specify '_target_' (e.g., 'torch.optim.AdamW')")

    # Filter out the target key, pass the rest as kwargs
    optimizer_kwargs = {k: v for k, v in optim_cfg.items() if k != "_target_"}

    logger.info(f"Creating optimizer: {optimizer_name} with params: {optimizer_kwargs}")

    try:
        # Currently only supporting AdamW explicitly for demonstration
        if optimizer_name.lower() == "torch.optim.adamw" or optimizer_name.lower() == "adamw":
            # Ensure required parameters are present (example: lr)
            if 'lr' not in optimizer_kwargs:
                raise ValueError("AdamW configuration must include 'lr' (learning_rate)")
            
            # Filter model parameters that require gradients
            params_to_optimize = [p for p in model.parameters() if p.requires_grad]
            logger.info(f"Optimizing {len(params_to_optimize)} parameter groups.")

            optimizer = optim.AdamW(params_to_optimize, **optimizer_kwargs)
        else:
            # Placeholder for potentially using hydra.utils.instantiate later
            # or adding more explicit optimizer support
            logger.warning(f"Optimizer '{optimizer_name}' not explicitly handled. Attempting general instantiation (may fail). Consider adding specific support.")
            # For now, raise error if not AdamW
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}. Currently only 'AdamW' is explicitly supported.")
            
            # Example using hydra (needs careful handling of parameter groups):
            # try:
            #     # This needs adjustment to correctly pass model params
            #     optimizer = hydra.utils.instantiate(optim_cfg, params=model.parameters())
            # except Exception as e:
            #     logger.error(f"Failed to instantiate optimizer {optimizer_name} using Hydra: {e}")
            #     raise

    except TypeError as e:
        logger.error(f"TypeError during optimizer instantiation for {optimizer_name}: {e}. Check config parameters.")
        raise ValueError(f"Invalid parameters for optimizer {optimizer_name}: {e}") from e
    except Exception as e:
        logger.error(f"Failed to create optimizer {optimizer_name}: {e}")
        raise

    logger.info(f"Optimizer {type(optimizer).__name__} created successfully.")
    return optimizer 