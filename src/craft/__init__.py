"""
Craft: A Modular Framework for AI Model Development

This package provides building blocks and orchestration tools for configuring,
building, training, and evaluating AI models, primarily using PyTorch and Hydra.
"""

__version__ = "1.0.0"
__author__ = "April Labs"

# Expose key components for easier top-level access.
# Users are still encouraged to import directly from submodules for clarity.

from .training.trainer import Trainer
from .config.schemas import (
    AppConfig,
    ExperimentConfig,
    TrainingConfig,
    DataConfig,
    AnyModelConfig
    # Add other core schemas if desired
)
from .utils.common import set_seed, setup_device
from .data.base import BaseDataset

# Removed obsolete or internal items from __all__
__all__ = [
    # Core Classes & Configs
    "Trainer",
    "AppConfig",
    "ExperimentConfig",
    "TrainingConfig",
    "DataConfig",
    "AnyModelConfig",
    "BaseDataset",

    # Key Utilities
    "set_seed",
    "setup_device",

    # Versioning/Info
    "__version__",
    "__author__",
] 