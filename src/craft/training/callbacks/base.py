"""
Base Callback Classes Module
===========================

Contains the abstract Callback base class and the CallbackList container.
"""

import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Abstract base class for callbacks."""
    def __init__(self):
        self.trainer = None # Initialize trainer attribute
        # Get a logger specific to the callback subclass
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_trainer(self, trainer):
        """Set the trainer instance for this callback."""
        self.trainer = trainer

    @abstractmethod
    def on_train_begin(self, trainer, logs=None):
        pass

    @abstractmethod
    def on_train_end(self, trainer, logs=None):
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer, epoch, logs=None):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer, epoch, logs=None):
        pass

    @abstractmethod
    def on_step_begin(self, step: int, logs=None):
        pass

    @abstractmethod
    def on_step_end(self, step: int, logs=None):
        pass

class CallbackList:
    """Container for managing a list of callbacks."""
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks if callbacks is not None else []
        self.trainer = None # Add trainer attribute

    def set_trainer(self, trainer):
        """Set the trainer instance for this list and all contained callbacks."""
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training begins."""
        if not self.trainer:
            logger.warning("CallbackList.on_train_begin called before trainer was set.")
            return # Or raise error
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(self.trainer, logs=logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called when training ends."""
        if not self.trainer:
            logger.warning("CallbackList.on_train_end called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(self.trainer, logs=logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        if not self.trainer:
            logger.warning("CallbackList.on_epoch_begin called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(self.trainer, epoch, logs=logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        if not self.trainer:
            logger.warning("CallbackList.on_epoch_end called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(self.trainer, epoch, logs=logs)

    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each training step."""
        if not self.trainer:
            logger.warning("CallbackList.on_step_begin called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_begin(step, logs=logs)

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each training step."""
        if not self.trainer:
            logger.warning("CallbackList.on_step_end called before trainer was set.")
            return
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_step_end(step, logs=logs)

    def append(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def extend(self, callbacks: List[Callback]):
        """Add multiple callbacks to the list."""
        self.callbacks.extend(callbacks) 