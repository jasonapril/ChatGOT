"""
Checkpointing Module
==================

This module handles saving and loading model checkpoints, including state management
and configuration serialization.
"""

import os
import torch
import logging
import shutil
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf

class CheckpointManager:
    """Manages saving and loading model checkpoints."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_checkpoint(
        self,
        path: str,
        current_epoch: int,
        global_step: int,
        best_val_metric: float,
        metrics: Dict[str, list],
        is_best: bool = False
    ) -> None:
        """Saves a checkpoint of the model and training state."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Ensure config is serializable (convert OmegaConf if needed)
        serializable_config = self.config
        try:
            # Attempt to resolve OmegaConf to primitive types if it's OmegaConf
            # Check if it's DictConfig rather than dict
            if isinstance(self.config, DictConfig):
                serializable_config = OmegaConf.to_container(self.config, resolve=True)
        except ImportError:
            # If OmegaConf not installed or fails, proceed with original (might raise error later)
            pass
        except Exception as e:
            self.logger.warning(f"Could not serialize OmegaConf config for checkpoint: {e}")
            # Fallback or store None?
            serializable_config = None # Avoid saving potentially problematic object

        checkpoint = {
            'epoch': current_epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'metrics': metrics,
            'config': serializable_config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        try:
            torch.save(checkpoint, path)
            self.logger.info(f"Checkpoint saved successfully to {path}")
            
            if is_best:
                # Optionally create a symlink or copy for 'best_model.pt'
                best_path = os.path.join(os.path.dirname(path), 'best_model.pt')
                # Simple copy for cross-platform compatibility
                # Add check to prevent copying onto itself
                if os.path.abspath(path) != os.path.abspath(best_path):
                    shutil.copyfile(path, best_path)
                    self.logger.info(f"Updated best model link to {best_path}")
                else:
                    # Log that we are saving directly as best_model.pt
                    self.logger.info(f"Saved best model directly as {best_path}")

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {path}: {e}", exc_info=True)

    def load_checkpoint(self, path: str) -> Optional[Dict[str, Any]]:
        """Loads the trainer state from a checkpoint file.

        Returns:
            The loaded config dictionary from the checkpoint, or None if loading failed.
        """
        if not os.path.exists(path):
            self.logger.error(f"Checkpoint file not found: {path}")
            return None

        loaded_config = None
        try:
            # Load checkpoint onto the correct device directly
            checkpoint = torch.load(path, map_location=self.model.device)

            # Load model state
            if 'model_state_dict' in checkpoint:
                # Handle potential DataParallel/DDP wrapping
                state_dict = checkpoint['model_state_dict']
                # Simple check for keys starting with 'module.'
                if any(key.startswith('module.') for key in state_dict.keys()):
                    self.logger.info("Detected 'module.' prefix in checkpoint state_dict, attempting to load into unwrapped model.")
                    # Create a new state_dict without the prefix
                    new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                    self.model.load_state_dict(new_state_dict)
                else:
                    self.model.load_state_dict(state_dict)
            else:
                self.logger.warning("Checkpoint does not contain 'model_state_dict'.")

            # Load optimizer state
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.logger.warning("Checkpoint does not contain 'optimizer_state_dict'. Optimizer state not loaded.")

            # Load scheduler state
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif self.scheduler:
                self.logger.warning("Checkpoint does not contain 'scheduler_state_dict'. Scheduler state not loaded.")

            # Load scaler state for AMP
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            elif self.scaler:
                self.logger.warning("Checkpoint does not contain 'scaler_state_dict'. AMP scaler state not loaded.")

            loaded_config = checkpoint.get('config')

            self.logger.info(f"Successfully loaded checkpoint from {path}")
            return loaded_config

        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found during load attempt: {path}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {path}: {e}", exc_info=True)
            return None 