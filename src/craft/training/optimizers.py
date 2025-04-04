"""
Factory function for creating optimizers.
"""
import logging
from typing import Dict, Any, Iterable

import torch
import torch.optim as optim
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def create_optimizer(model, optimizer_cfg):
    """
    Creates an optimizer based on the provided configuration.

    Args:
        model: The model whose parameters need optimization.
        optimizer_cfg: The OmegaConf DictConfig object for the optimizer, 
                   expected to have '_target_' and parameters like 'lr'.

    Returns:
        An instantiated PyTorch optimizer.

    Raises:
        ValueError: If the optimizer type specified by '_target_' is unknown 
                    or if required parameters are missing.
    """
    # --- DEBUGGING --- 
    # logger.debug(f"[DEBUG][Optimizer] Received optimizer_cfg type: {type(optimizer_cfg)}")
    # logger.debug(f"[DEBUG][Optimizer] Received optimizer_cfg content: {optimizer_cfg}")
    # logger.debug(f"[DEBUG][Optimizer] optimizer_cfg.get('target'): {optimizer_cfg.get('target')}")
    # logger.debug(f"[DEBUG][Optimizer] optimizer_cfg.get('_target_'): {optimizer_cfg.get('_target_')}")
    # --- END DEBUGGING ---

    # Ensure target key is present (check both 'target' and '_target_')
    target_key_alias = "target"
    target_key_hydra = "_target_"
    target_path = optimizer_cfg.get(target_key_alias) or optimizer_cfg.get(target_key_hydra)
    if not target_path:
        raise ValueError(f"Optimizer configuration must specify '{target_key_alias}' or '{target_key_hydra}'.")

    # Filter out the target key(s), pass the rest as kwargs
    optimizer_kwargs = optimizer_cfg.copy() # Make a copy
    # Remove both potential keys
    optimizer_kwargs.pop(target_key_alias, None)
    optimizer_kwargs.pop(target_key_hydra, None)

    logger.info(f"Creating optimizer: {target_path} with params: {optimizer_kwargs}")

    try:
        # Currently only supporting AdamW explicitly for demonstration
        if target_path.lower() == "torch.optim.adamw" or target_path.lower() == "adamw":
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
            logger.warning(f"Optimizer '{target_path}' not explicitly handled. Attempting general instantiation (may fail). Consider adding specific support.")
            # For now, raise error if not AdamW
            raise ValueError(f"Unsupported optimizer type: {target_path}. Currently only 'AdamW' is explicitly supported.")
            
            # Example using hydra (needs careful handling of parameter groups):
            # try:
            #     # This needs adjustment to correctly pass model params
            #     optimizer = hydra.utils.instantiate(optimizer_cfg, params=model.parameters())
            # except Exception as e:
            #     logger.error(f"Failed to instantiate optimizer {target_path} using Hydra: {e}")
            #     raise

    except TypeError as e:
        logger.error(f"TypeError during optimizer instantiation for {target_path}: {e}. Check config parameters.")
        raise ValueError(f"Invalid parameters for optimizer {target_path}: {e}") from e
    except Exception as e:
        logger.error(f"Failed to create optimizer {target_path}: {e}", exc_info=True)
        raise

    logger.info(f"Optimizer {type(optimizer).__name__} created successfully.")
    return optimizer 