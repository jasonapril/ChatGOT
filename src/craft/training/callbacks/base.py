"""
Base Callback Classes Module
===========================

Contains the abstract Callback base class and the CallbackList container.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING, Iterator
from abc import ABC, abstractmethod

# Added TYPE_CHECKING block for forward references
if TYPE_CHECKING:
    from ..trainer import Trainer  # Assuming Trainer is in trainer.py
    from ..checkpointing import TrainingState # Corrected import from checkpointing.py

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Abstract base class for callbacks."""
    def __init__(self) -> None:
        # Type hint for trainer uses forward reference string implicitly
        self.trainer: Optional['Trainer'] = None # Will be set by Trainer
        # Get a logger specific to the callback subclass
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_trainer(self, trainer: 'Trainer') -> None:
        """Sets the trainer instance for the callback."""
        self.trainer = trainer

    def on_init_end(self, **kwargs: Any) -> None:
        """Called at the end of Trainer.__init__."""
        pass

    def on_train_begin(self, **kwargs: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, metrics: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Called at the end of an epoch."""
        pass

    def on_step_begin(self, step: int, **kwargs: Any) -> None:
        """Called at the beginning of a training step."""
        pass

    def on_step_end(self, step: int, global_step: int, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Called at the end of a training step."""
        pass

    def on_evaluation_begin(self, **kwargs: Any) -> None:
        """Called before evaluation starts."""
        pass

    def on_evaluation_end(self, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Called after evaluation ends."""
        pass

    def on_save_checkpoint(self, state: 'TrainingState', filename: str, **kwargs: Any) -> None:
        """Called when a checkpoint is saved."""
        pass

    def on_load_checkpoint(self, state: 'TrainingState', filename: str, **kwargs: Any) -> None:
        """Called when a checkpoint is loaded."""
        pass

    def on_exception(self, exception: Exception, **kwargs: Any) -> None:
        """Called when an exception occurs during training."""
        pass

class CallbackList:
    """Manages a list of callbacks."""
    def __init__(self, callbacks: Optional[List[Callback]] = None, *, trainer: Optional['Trainer'] = None) -> None:
        self.callbacks = callbacks if callbacks else []
        self.trainer = trainer # Store trainer if provided during init
        # Set trainer for existing callbacks if provided
        if self.trainer:
            self.set_trainer(self.trainer)

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def set_trainer(self, trainer: 'Trainer') -> None:
        """Set the trainer instance for this list and all contained callbacks."""
        self.trainer = trainer # Initialize self.trainer here
        for callback in self.callbacks:
            if isinstance(callback, Callback):
                callback.set_trainer(trainer)
            else:
                # Log a warning or handle non-Callback objects appropriately
                if hasattr(callback, 'set_trainer') and callable(getattr(callback, 'set_trainer')):
                   callback.set_trainer(trainer)
                else:
                   # Use module-level logger instead of self.logger
                   logger.warning(f"Item in CallbackList ({type(callback)}) does not have a callable set_trainer method. Skipping.")

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
            raise AttributeError(f"'{self.__class__.__name__}\' object has no attribute '{name}' or it's not a valid callback method")

        def method(*args: Any, **kwargs: Any) -> None:
            # Ensure trainer is set before dispatching calls that might need it
            # It's the responsibility of the caller (Trainer) to ensure set_trainer was called.
            # We could add a check here, but it might add overhead.
            # if not hasattr(self, 'trainer') or self.trainer is None:
            #     logger.warning(f"CallbackList.{name} called before trainer was set.")
            #     # Depending on the method, we might need to raise an error or return early.

            for callback in self.callbacks:
                # Check if the callback instance has the method before calling
                if hasattr(callback, name) and callable(getattr(callback, name)):
                    try:
                        # Forward all args and kwargs
                        getattr(callback, name)(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in callback {callback.__class__.__name__}.{name}: {e}", exc_info=True) # Added exc_info
                        # Optionally re-raise or handle differently
                        # raise e # Re-raise to stop execution if needed

        return method

    # Removed explicit on_epoch_begin method
    # Removed explicit on_train_begin method
    # Removed explicit on_train_end method
    # Removed explicit on_epoch_end method
    # Removed explicit on_step_begin method
    # Removed explicit on_step_end method

    def append(self, callback: Callback) -> None:
        """Adds a callback to the list."""
        self.callbacks.append(callback)

    def state_dict(self) -> Dict[str, Any]:
        # Implementation of state_dict method needs to be added
        # Should likely aggregate state_dicts from individual callbacks that have them
        logger.warning("CallbackList.state_dict is not fully implemented.")
        return {"callbacks": [cb.__class__.__name__ for cb in self.callbacks]} # Placeholder

    def __iter__(self) -> Iterator[Callback]:
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks) 