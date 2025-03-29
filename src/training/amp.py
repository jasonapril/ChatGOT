"""
Custom utilities for Automatic Mixed Precision (AMP) training.
"""

import logging
import torch
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)

class SafeGradScaler(GradScaler):
    """
    Enhanced GradScaler with NaN detection and fallback to full precision.

    Inherits from torch.cuda.amp.GradScaler and adds:
    - Checks for NaN/Inf in the loss before scaling.
    - If NaN/Inf occurs repeatedly, disables AMP and falls back to full precision.
    - Checks for NaN/Inf during the optimizer step and attempts fallback.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize SafeGradScaler.

        Args:
            *args: Positional arguments passed to GradScaler.
            **kwargs: Keyword arguments passed to GradScaler.
                      Includes an additional parameter:
                      max_nan_before_fallback (int): Max consecutive NaN/Inf occurrences
                                                     before disabling AMP (default: 3).
        """
        self.max_nan_before_fallback = kwargs.pop('max_nan_before_fallback', 3)
        super().__init__(*args, **kwargs)
        self.nan_counter = 0
        self.fallback_triggered = False
        self.warning_logged = False # Track if fallback warning was logged

    def scale(self, loss):
        """
        Scales the loss tensor. Checks for NaN/Inf before scaling.
        If fallback is triggered or AMP is disabled, returns the unscaled loss.
        """
        # Check for NaN/Inf before scaling
        if not self._enabled:
            # If AMP is already disabled (manually or by fallback), return unscaled loss
            if not self.warning_logged:
                 logger.warning("Mixed precision is disabled. Loss scaling skipped.")
                 self.warning_logged = True
            return loss

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self.nan_counter += 1
            loss_val_str = f"{loss.item() if not torch.isinf(loss).all() else 'inf'}"
            logger.warning(
                f"NaN/Inf detected in loss before scaling: {loss_val_str} "
                f"(occurrence {self.nan_counter}/{self.max_nan_before_fallback})"
            )

            # If we've seen too many NaNs, disable mixed precision permanently
            if self.nan_counter >= self.max_nan_before_fallback and not self.fallback_triggered:
                logger.error(
                    f"Disabling mixed precision permanently due to repeated NaN/Inf values in loss."
                )
                self._enabled = False # Disable scaler
                self.fallback_triggered = True
                self.warning_logged = True # Ensure warning is logged on next call

            # Return unscaled loss to potentially continue training (caller should check)
            # Or, if fallback triggered, we are now in full precision mode anyway.
            return loss
        else:
            # Reset counter slightly if loss is normal (prevents single spikes disabling AMP)
            if self.nan_counter > 0:
                self.nan_counter = max(0, self.nan_counter - 1) # Gradually decrease counter

        # If we reach here, AMP is enabled and loss is valid, proceed with scaling
        return super().scale(loss)

    def step(self, optimizer, *args, **kwargs):
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
                self.fallback_triggered = True
                self.warning_logged = True

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

    def update(self, new_scale=None):
        """
        Updates the scale factor. Skips if AMP is disabled.
        """
        if not self._enabled:
            return # Do nothing if fallback is active
        super().update(new_scale=new_scale)

    def state_dict(self):
        """Returns the state dict, adding fallback status."""
        state_dict = super().state_dict()
        state_dict["fallback_triggered"] = self.fallback_triggered
        state_dict["nan_counter"] = self.nan_counter
        # Note: _enabled state is implicitly saved via scale factor etc.
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the state dict, restoring fallback status."""
        self.fallback_triggered = state_dict.get("fallback_triggered", False)
        self.nan_counter = state_dict.get("nan_counter", 0)

        # If fallback was triggered, ensure the scaler remains disabled
        if self.fallback_triggered:
             self._enabled = False
             self.warning_logged = True # Assume warning is needed if loading a fallback state

        super().load_state_dict(state_dict)
        # Re-check enabled status after loading GradScaler's state
        if self.fallback_triggered:
             self._enabled = False

# </rewritten_file> 