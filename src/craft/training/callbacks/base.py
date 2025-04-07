"""
Base Callback Classes Module
===========================

Contains the abstract Callback base class and the CallbackList container.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Abstract base class for callbacks."""
    def __init__(self):
        self.trainer: Optional['Trainer'] = None # Will be set by Trainer
        # Get a logger specific to the callback subclass
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_trainer(self, trainer: 'Trainer'):
        """Sets the trainer instance for the callback."""
        self.trainer = trainer

    def on_init_end(self, **kwargs):
        """Called at the end of Trainer.__init__."""
        pass

    def on_train_begin(self, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, global_step: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of an epoch."""
        pass

    def on_step_begin(self, step: int, **kwargs):
        """Called at the beginning of a training step."""
        pass

    def on_step_end(self, step: int, global_step: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of a training step."""
        pass

    def on_evaluation_begin(self, **kwargs):
        """Called before evaluation starts."""
        pass

    def on_evaluation_end(self, metrics: Dict[str, Any], **kwargs):
        """Called after evaluation ends."""
        pass

    def on_save_checkpoint(self, state: 'TrainingState', filename: str, **kwargs):
        """Called when a checkpoint is saved."""
        pass

    def on_load_checkpoint(self, state: 'TrainingState', filename: str, **kwargs):
        """Called when a checkpoint is loaded."""
        pass

    def on_exception(self, exception: Exception, **kwargs):
        """Called when an exception occurs during training."""
        pass

class CallbackList:
    """Manages a list of callbacks."""
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks if callbacks else []
        self.trainer = None # Add trainer attribute

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def set_trainer(self, trainer: 'Trainer'):
        """Set the trainer instance for this list and all contained callbacks."""
        self.trainer = trainer
        for callback in self.callbacks:
            if isinstance(callback, Callback):
                callback.set_trainer(trainer)
            else:
                # Log a warning or handle non-Callback objects appropriately
                if hasattr(callback, 'set_trainer') and callable(getattr(callback, 'set_trainer')):
                   callback.set_trainer(trainer)
                else:
                   self.logger.warning(f"Item in CallbackList ({type(callback)}) does not have a callable set_trainer method. Skipping.")

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Dynamically call the corresponding method on all callbacks."""
        # Check if the requested attribute is a known callback method
        known_methods = [
            'on_init_end', 'on_train_begin', 'on_train_end', 'on_epoch_begin',
            'on_epoch_end', 'on_step_begin', 'on_step_end', 'on_evaluation_begin',
            'on_evaluation_end', 'on_save_checkpoint', 'on_load_checkpoint',
            'on_exception'
        ]
        if name not in known_methods:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' or it's not a valid callback method")

        def method(*args, **kwargs):
            for callback in self.callbacks:
                # Check if the callback instance has the method before calling
                if hasattr(callback, name) and callable(getattr(callback, name)):
                    try:
                        getattr(callback, name)(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in callback {callback.__class__.__name__}.{name}: {e}")
                        # Optionally re-raise or handle differently
                        # raise e # Re-raise to stop execution if needed

        return method

    def on_epoch_begin(self, trainer: 'Trainer', current_epoch: int, global_step: int, **kwargs):
        """Calls on_epoch_begin on all contained callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin') and callable(getattr(callback, 'on_epoch_begin')):
                try:
                    # Pass trainer, current_epoch, global_step, and any other kwargs
                    callback.on_epoch_begin(trainer=trainer, current_epoch=current_epoch, global_step=global_step, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__class__.__name__}.on_epoch_begin: {e}", exc_info=True)

    # --- Specific method for on_train_begin (overrides __getattr__ for this method) ---
    def on_train_begin(self, **kwargs):
        """Called when training begins. Passes kwargs to individual callbacks."""
        if not self.trainer:
            logger.warning("CallbackList.on_train_begin called before trainer was set.")
            # Decide if we should return or raise an error here.
            # Returning might silently fail later if callbacks rely on the trainer.
            # Raising an error might be safer.
            # For now, just log and continue.
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin') and callable(getattr(callback, 'on_train_begin')):
                try:
                    # Pass any kwargs received by CallbackList down to the individual callback
                    callback.on_train_begin(**kwargs)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__class__.__name__}.on_train_begin: {e}")
    # --- End specific method ---

    def on_train_end(self, metrics: Optional[Dict[str, Any]] = None, **kwargs):
        """Called when training ends."""
        if not self.trainer:
            logger.warning("CallbackList.on_train_end called before trainer was set.")
            return
        metrics = metrics or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end') and callable(getattr(callback, 'on_train_end')):
                try:
                     # Pass metrics dict and any other kwargs
                    callback.on_train_end(metrics=metrics, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__class__.__name__}.on_train_end: {e}", exc_info=True)

    def on_epoch_end(self, epoch: int, global_step: int, metrics: Dict[str, Any], **kwargs):
        """Called at the end of each epoch."""
        if not self.trainer:
            logger.warning("CallbackList.on_epoch_end called before trainer was set.")
            return
        metrics = metrics or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end') and callable(getattr(callback, 'on_epoch_end')):
                try:
                    callback.on_epoch_end(epoch=epoch, global_step=global_step, metrics=metrics, **kwargs)
                except Exception as e:
                     logger.error(f"Error in callback {callback.__class__.__name__}.on_epoch_end: {e}", exc_info=True)

    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each step."""
        if not self.trainer:
            logger.warning("CallbackList.on_step_begin called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_begin(step, **logs)

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each step."""
        if not self.trainer:
            logger.warning("CallbackList.on_step_end called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_end(step, **logs)

    def append(self, callback: Callback):
        """Adds a callback to the list."""
        self.callbacks.append(callback)

    def state_dict(self):
        # Implementation of state_dict method
        pass

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self):
        return len(self.callbacks) 