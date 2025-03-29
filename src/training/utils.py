"""
Training utilities, including gradient checkpointing.
"""

import logging
import functools
import torch

# Try to import checkpoint functionality from the correct location based on PyTorch version
try:
    # Newer PyTorch versions (>=1.10 ? or earlier?)
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    if not callable(torch_checkpoint): # Handle potential import issues or changes
         raise ImportError("torch.utils.checkpoint.checkpoint is not callable")
except ImportError:
    # Older PyTorch versions might have it here
    try:
        # Check if checkpoint exists and is callable directly in torch.utils
        from torch.utils import checkpoint as torch_checkpoint_module
        if hasattr(torch_checkpoint_module, 'checkpoint') and callable(torch_checkpoint_module.checkpoint):
             torch_checkpoint = torch_checkpoint_module.checkpoint
        else:
             # Maybe it's directly checkpoint? Less common.
             from torch.utils import checkpoint as torch_checkpoint
             if not callable(torch_checkpoint):
                 raise ImportError("Could not find callable checkpoint function in torch.utils")
    except ImportError:
        logging.getLogger(__name__).warning(
            "Could not import torch.utils.checkpoint. Gradient checkpointing will not be available."
        )
        torch_checkpoint = None

def enable_gradient_checkpointing(model: torch.nn.Module):
    """
    Safely enable gradient checkpointing on model layers.

    Assumes the model has a 'layers' attribute containing the sequence of layers
    to apply checkpointing to. Modifies the `forward` method of each layer.

    Args:
        model: The model to enable gradient checkpointing on.

    Returns:
        bool: True if checkpointing was successfully enabled, False otherwise.
    """
    logger = logging.getLogger(__name__)

    if torch_checkpoint is None:
        logger.warning("torch.utils.checkpoint not found. Cannot enable gradient checkpointing.")
        return False

    # TODO: Make layer access more robust (e.g., check common attributes like model.transformer.h)
    if not hasattr(model, 'layers'):
        logger.warning("Model doesn't have a 'layers' attribute. Skipping gradient checkpointing.")
        return False

    try:
        logger.info("Attempting to enable gradient checkpointing...")
        count = 0
        for i, layer in enumerate(model.layers):
            if not hasattr(layer, 'forward_original'):
                layer.forward_original = layer.forward

                # Create a wrapper for checkpointing
                def checkpoint_wrapper(original_forward, *args, **kwargs):
                    # Filter out kwargs not accepted by torch.utils.checkpoint if necessary
                    # Depending on the PyTorch version, non-tensor kwargs might cause issues.
                    # For simplicity now, assume args are positional tensors and kwargs are compatible.
                    try:
                        # Ensure use_reentrant=False for better memory efficiency in newer PyTorch
                        # This might require PyTorch >= 1.10 or later. Adjust based on target env.
                        kwargs_for_checkpoint = {'use_reentrant': False}
                        # If use_reentrant=False is not available or causes issues, remove it
                        # return torch_checkpoint(original_forward, *args, **kwargs) # Older way
                        return torch_checkpoint(original_forward, *args, **kwargs_for_checkpoint)
                    except TypeError as te:
                        if 'use_reentrant' in str(te):
                             logger.warning("`use_reentrant=False` not supported or caused error. Falling back.")
                             return torch_checkpoint(original_forward, *args, **kwargs)
                        else:
                             raise # Re-raise other TypeErrors
                    except Exception as e:
                        inner_logger = logging.getLogger(__name__)
                        inner_logger.error(f"Error during checkpointed forward pass in layer {i}: {e}", exc_info=True)
                        # Fall back to original forward pass if checkpointing fails at runtime
                        return original_forward(*args, **kwargs)

                layer.forward = functools.partial(checkpoint_wrapper, layer.forward_original)
                # logger.debug(f"Gradient checkpointing enabled for layer {i}") # Use debug level
                count += 1
            else:
                 logger.warning(f"Layer {i} already seems to have checkpointing enabled or 'forward_original' exists.")

        if count > 0:
             logger.info(f"Gradient checkpointing enabled for {count} layers.")
        else:
             logger.warning("No layers were modified for gradient checkpointing (potentially already enabled or model structure issue).")
        return True
    except Exception as e:
        logger.error(f"Failed to enable gradient checkpointing: {e}", exc_info=True)
        # Attempt to disable if setup failed midway
        disable_gradient_checkpointing(model)
        return False

def disable_gradient_checkpointing(model: torch.nn.Module):
    """
    Safely disable gradient checkpointing and restore original forward functions.

    Args:
        model: The model to disable gradient checkpointing on.

    Returns:
        bool: True if checkpointing was successfully disabled, False otherwise.
    """
    logger = logging.getLogger(__name__)

    if not hasattr(model, 'layers'):
        # If the model structure doesn't match, nothing to disable
        return True

    try:
        logger.info("Attempting to disable gradient checkpointing...")
        count = 0
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'forward_original'):
                layer.forward = layer.forward_original
                delattr(layer, 'forward_original')
                # logger.debug(f"Gradient checkpointing disabled for layer {i}") # Use debug level
                count += 1

        if count > 0:
            logger.info(f"Gradient checkpointing disabled for {count} layers.")
        else:
            logger.info("No layers found with gradient checkpointing enabled.")
        return True
    except Exception as e:
        logger.error(f"Error disabling gradient checkpointing: {e}", exc_info=True)
        return False 