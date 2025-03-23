"""
ChatGoT: Character-Level Transformer for Game of Thrones Text Generation
========================================================================

This package implements a character-level transformer model for generating
Game of Thrones style text, optimized for NVIDIA GPUs.
"""

__version__ = "1.0.0"
__author__ = "ChatGoT Team"

# Import key components for easier access
from .model import create_transformer_model, TransformerModel
from .trainer import train_epoch, evaluate, generate_text, sample_text
from .data_handler import load_data
from .memory_management import get_memory_optimized_settings, preallocate_gpu_memory
from .logger import setup_logger, log_section_header, force_flush_logs, format_time
from .utils import (
    set_seed,
    setup_device,
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    create_output_dir,
    save_args
) 