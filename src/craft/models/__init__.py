"""
Models module initialization.
Imports concrete model implementations and exposes base classes.
"""
import logging

# Import base classes first
from .base import Model, GenerativeModel, LanguageModel, VisionModel, MultiModalModel

# Import concrete model implementations
from .simple_rnn import SimpleRNN
from .transformer import TransformerModel
# Add imports for other model modules here (e.g., .vit, .clip, etc.)

# Configs are now imported directly from craft.config.schemas where needed
# Factory and registry are removed.

# Import from the central config schemas location
from craft.config.schemas import BaseModelConfig

# Setup logger
logger = logging.getLogger(__name__)
logger.debug("Models module initialized.")

# Define __all__ for explicit exports from this package level
__all__ = [
    # Base Classes
    "Model",
    "GenerativeModel",
    "LanguageModel",
    "VisionModel",
    "MultiModalModel",
    "BaseModelConfig",

    # Concrete Models (Re-exported for easier access if desired)
    "SimpleRNN",
    "TransformerModel",
    # Add other model class names here
] 