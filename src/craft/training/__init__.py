"""
Craft Training Subsystem Initialization

Exposes key components for training, evaluation, generation, and utilities.
"""

from .trainer import Trainer
from .training_loop import TrainingLoop
from .evaluation import Evaluator
from .checkpointing import CheckpointManager
from .callbacks import Callback, CallbackList
from .progress import ProgressTracker
# Import from generation modules
from .generation import TextGenerator # Main generator class
from .sampling import generate_text_manual_sampling, generate_samples_manual_sampling # Renamed functions

__all__ = [
    # Core Training Classes
    "Trainer",
    "TrainingLoop",
    "Evaluator",
    "CheckpointManager",

    # Callbacks
    "Callback",
    "CallbackList",

    # Utilities
    "ProgressTracker",
    "TextGenerator",

    # Specific Generation Functions (Optional to expose)
    "generate_text_manual_sampling",
    "generate_samples_manual_sampling",
] 