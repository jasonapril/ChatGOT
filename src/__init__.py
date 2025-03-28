"""
Character-Level Transformer for Text Generation

A framework for experimenting with language models and AI architectures.
"""

__version__ = "1.0.0"
__author__ = "April Labs"

# Import key components for easier access
from .models.transformer import create_transformer_model, TransformerModel
from .utils.generation import generate_sample_text, sample_text
from .data.dataset import CharDataset, load_data
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