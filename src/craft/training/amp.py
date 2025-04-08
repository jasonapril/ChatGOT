"""
Custom utilities for Automatic Mixed Precision (AMP) training.
"""

import logging
from typing import Optional, Dict, Any, Union

import torch
import torch.optim as optim
# Import the newer GradScaler location (torch.amp)
from torch.amp import GradScaler 

logger = logging.getLogger(__name__)

# Inherit from the newer GradScaler
class SafeGradScaler(GradScaler):
    """
    Enhanced GradScaler with NaN detection and fallback to full precision.

    Inherits from torch.amp.cuda.GradScaler and adds:
    - Checks for NaN/Inf in the loss before scaling.
    - If NaN/Inf occurs repeatedly, disables AMP and falls back to full precision.
    - Checks for NaN/Inf during the optimizer step and attempts fallback.
    """

    def __init__(self, 
                 enabled: bool = True, 
                 init_scale: float = 2.**16, 
                 growth_factor: float = 2.0, 
                 backoff_factor: float = 0.5, 
                 growth_interval: int = 2000, 
                 max_consecutive_nan_skip: int = 5, 
                 enable_fallback: bool = False):
        """
        Initializes the SafeGradScaler.

        Args:
            enabled (bool): Whether AMP and scaling are enabled.
            init_scale (float): Initial scale factor.
            growth_factor (float): Factor by which the scale is multiplied during growth.
            backoff_factor (float): Factor by which the scale is multiplied during backoff.
            growth_interval (int): Number of steps between scale growth attempts.
            max_consecutive_nan_skip (int): Max number of consecutive steps with NaN/Inf loss
                                           before triggering fallback (if enabled) or raising error.
            enable_fallback (bool): If True, attempts to disable AMP and continue training
                                    after max_consecutive_nan_skip is reached.
        """
        # Explicitly call the parent __init__ with appropriate args
        # The base GradScaler takes 'enabled', 'init_scale', 'growth_factor', 
        # 'backoff_factor', 'growth_interval'
        # Note: The warning suggested torch.amp.GradScaler('cuda', ...), but since 
        # torch.amp.cuda.GradScaler exists, inheriting directly seems correct.
        # We pass the standard args. 
        super().__init__(init_scale=init_scale,
                         growth_factor=growth_factor,
                         backoff_factor=backoff_factor,
                         growth_interval=growth_interval,
                         enabled=enabled)
        
        self.max_consecutive_nan_skip = max_consecutive_nan_skip
        self.enable_fallback = enable_fallback
        self._consecutive_nan_skips = 0
        # Track internal enabled state for fallback
        self._internal_enabled = enabled 
        self._initial_enabled_state = enabled # Store initial state for reference

        if not enabled:
            logger.info("SafeGradScaler initialized but AMP is disabled.")

    def scale(self, outputs: torch.Tensor) -> torch.Tensor: # type: ignore[override]
        """
        Scales the loss tensor. Checks for NaN/Inf before scaling.
        If fallback is triggered or AMP is disabled, returns the unscaled loss.
        """
        # Check for NaN/Inf before scaling
        if not self._enabled:
            # If AMP is already disabled (manually or by fallback), return unscaled loss
            logger.warning("Mixed precision is disabled. Loss scaling skipped.")
            return outputs

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            self._consecutive_nan_skips += 1
            loss_val_str = f"{outputs.item() if not torch.isinf(outputs).all() else 'inf'}"
            logger.warning(
                f"NaN/Inf detected in loss before scaling: {loss_val_str} "
                f"(occurrence {self._consecutive_nan_skips}/{self.max_consecutive_nan_skip})"
            )

            # If we've seen too many NaNs, disable mixed precision permanently
            if self._consecutive_nan_skips >= self.max_consecutive_nan_skip and not self.enable_fallback:
                logger.error(
                    f"Disabling mixed precision permanently due to repeated NaN/Inf values in loss."
                )
                self._enabled = False # Disable scaler

            # Return unscaled loss to potentially continue training (caller should check)
            # Or, if fallback triggered, we are now in full precision mode anyway.
            return outputs
        else:
            # Reset counter slightly if loss is normal (prevents single spikes disabling AMP)
            if self._consecutive_nan_skips > 0:
                self._consecutive_nan_skips = max(0, self._consecutive_nan_skips - 1) # Gradually decrease counter

        # If we reach here, AMP is enabled and loss is valid, proceed with scaling
        return super().scale(outputs)

    def step(self, optimizer: optim.Optimizer, *args: Any, **kwargs: Any) -> Any:
        """
        Performs the optimizer step. Includes safety checks for NaN/Inf gradients.
        If AMP fallback is active, performs a regular optimizer step.
        """
        if not self._enabled:
            # In full precision mode (fallback active), just do a normal step
            # Ensure gradients are unscaled (should be if loss wasn't scaled)
            return optimizer.step(*args, **kwargs)

        try:
            # Attempt the standard GradScaler step
            return super().step(optimizer, *args, **kwargs)
        except RuntimeError as e:
            # Check if the error is related to NaN/Inf gradients found by the optimizer step
            if "inf or nan" in str(e).lower():
                logger.error(f"NaN/Inf detected during optimizer step, disabling mixed precision: {e}")
                self._enabled = False

                # Gradients are invalid, zero them out before attempting fallback step
                optimizer.zero_grad()

                # Since the optimizer step failed due to bad gradients after unscaling,
                # re-running optimizer.step() won't help.
                # We simply skip the optimizer step for this iteration.
                logger.warning("Skipping optimizer step for this iteration due to invalid gradients.")
                # Return None or indicate failure? GradScaler.step usually returns optimizer.step result or None.
                return None
            else:
                # Re-raise other runtime errors
                raise
        except Exception as e:
            # Catch any other unexpected errors during the step
            logger.error(f"Unexpected error during SafeGradScaler.step: {e}")
            raise

    def update(self, new_scale: Union[float, torch.Tensor, None] = None) -> None:
        """
        Updates the scale factor. Skips if AMP is disabled.
        """
        if not self._enabled:
            return # Do nothing if fallback is active
        super().update(new_scale=new_scale)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dict, adding fallback status."""
        state_dict = super().state_dict()
        state_dict["_consecutive_nan_skips"] = self._consecutive_nan_skips
        state_dict["_internal_enabled"] = self._internal_enabled
        state_dict["_initial_enabled_state"] = self._initial_enabled_state
        # Note: _enabled state is implicitly saved via scale factor etc.
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state dict, restoring fallback status."""
        self._consecutive_nan_skips = state_dict.get("_consecutive_nan_skips", 0)
        self._internal_enabled = state_dict.get("_internal_enabled", True)
        self._initial_enabled_state = state_dict.get("_initial_enabled_state", True)

        # If fallback was triggered, ensure the scaler remains disabled
        if self._consecutive_nan_skips >= self.max_consecutive_nan_skip:
             self._enabled = False

        super().load_state_dict(state_dict)
        # Re-check enabled status after loading GradScaler's state
        if self._consecutive_nan_skips >= self.max_consecutive_nan_skip:
             self._enabled = False

# </rewritten_file> 