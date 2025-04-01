"""
Character-Level Transformer for Text Generation

A framework for experimenting with language models and AI architectures.
"""

__version__ = "1.0.0"
__author__ = "April Labs"

# Import key components for easier access
# from .models.transformer import create_transformer_model, TransformerModel # Old import
from .models.transformer import TransformerModel # Keep only TransformerModel if needed here
from .models import create_model_from_config # Import the factory function
from .utils.generation import generate_sample_text, sample_text
# Removed CharDataset import
from .utils.memory import get_memory_optimized_settings, preallocate_gpu_memory
from .utils.metrics import calculate_tokens_per_second, calculate_perplexity
from .utils.logging import setup_logger, log_section_header, force_flush_logs, format_time
from .utils.common import (
    set_seed,
    setup_device,
    get_memory_usage,
    format_time,
    format_number
) 
# from .config import settings # Removed import
from .utils.logging import setup_logging # Import only setup_logging if needed
# from .data.processors import prepare_data # Example function
# from .models.transformer import TransformerModel # Example model
# from .training.trainer import Trainer # Example trainer
from .data.base import create_dataset_from_config # Base dataset factory
from .data.dataset import PickledDataset # Current dataset implementation

# Initialize logging as soon as the package is imported
# setup_logging()

__all__ = [
    "create_model_from_config",
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