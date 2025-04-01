"""
Character-Level Transformer for Text Generation

A framework for experimenting with language models and AI architectures.
"""

__version__ = "1.0.0"
__author__ = "April Labs"

# Minimal imports - let users import from submodules directly
# Example: from craft.models import create_model_from_config
# Example: from craft.data import PickledDataset

# Maybe initialize logging here if desired system-wide
# from .utils.logging import setup_logging
# setup_logging()

__all__ = [
    # Data
    "BaseDataset",
    # "load_data", # Removed
    "DataManager",
    "create_data_manager_from_config",
    "create_dataset_from_config",
    # "settings", # Removed from __all__
    # "logger", # Removed from __all__
    # "prepare_data",
    "PickledDataset",
    # "TransformerModel",
    # "Trainer"
] 