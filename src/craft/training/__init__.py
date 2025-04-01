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
from .optimizers import create_optimizer
from .schedulers import create_scheduler
# Import from new generation modules
from .generation import TextGenerator # Wrapper class
from .sampling import generate_text_sampling, sample_text # Standalone sampling
from .beam_search import beam_search_generate # Standalone beam search
from .batch_generation import batch_generate # Standalone batch generation

__all__ = [
    "Trainer",
    "TrainingLoop",
    "Evaluator",
    "CheckpointManager",
    "Callback",
    "CallbackList",
    "ProgressTracker",
    "create_optimizer",
    "create_scheduler",
    "TextGenerator",
    "generate_text_sampling", # Renamed to avoid conflict
    "sample_text",
    "beam_search_generate",
    "batch_generate",
] 