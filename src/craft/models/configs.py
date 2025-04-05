"""
Model Configuration Schemas using Pydantic.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Type
from pydantic import BaseModel, Field, ConfigDict, field_validator
import logging

logger = logging.getLogger(__name__)

# --- Base Config --- #
class BaseModelConfig(BaseModel):
    """
    Base Pydantic configuration class for all models.
    Provides automatic validation and type hints.
    Uses model_config for Pydantic V2 settings.
    """
    model_config = ConfigDict(extra='allow') # Keep allow for flexibility
    model_type: str = Field("base", description="The type of the model (e.g., language, vision).")

# --- Generative Config --- #
class GenerativeModelConfig(BaseModelConfig):
    """Config for Generative Models"""
    model_type: str = Field("generative", description="Model type set to generative.")
    max_seq_length: int = Field(1024, description="Maximum sequence length the model can handle")

# --- Language Model Config --- #
class LanguageModelConfig(GenerativeModelConfig):
    """Config for Language Models"""
    model_type: str = Field("language", description="Model type set to language.")
    architecture: Optional[str] = Field(None, description="Name for the specific model architecture (e.g., transformer, rnn).")
    vocab_size: int = Field(..., description="Size of the vocabulary (required).")
    d_model: int = Field(768, description="Model dimension.")
    n_head: int = Field(12, description="Number of attention heads.")
    d_hid: Optional[int] = Field(None, description="Hidden dimension in feed-forward layers.")
    n_layers: int = Field(12, description="Number of transformer layers.")
    dropout: float = Field(0.1, description="Dropout probability.")
    bias: bool = Field(True, description="Whether to use bias in linear layers.")
    layer_norm_eps: float = Field(1e-5, description="Epsilon for layer normalization.")
    activation: str = Field('gelu', description="Activation function.")
    norm_first: bool = Field(True, description="Apply layer norm before attention/FFN.")

    @field_validator('d_hid', mode='before')
    @classmethod
    def set_d_hid_default(cls, v, info):
        """Set default d_hid = d_model * 4 if not provided."""
        if v is None and 'd_model' in info.data:
            d_model = info.data.get('d_model')
            if d_model is None and 'd_model' in cls.model_fields:
                 d_model = cls.model_fields['d_model'].default
            if isinstance(d_model, int):
                return d_model * 4
        return v

# --- Vision Model Config --- #
class VisionModelConfig(BaseModelConfig):
    """Placeholder Config for Vision Models"""
    model_type: str = Field("vision", description="Model type set to vision.")
    image_size: Tuple[int, int] = Field((224, 224), description="Input image dimensions (height, width).")
    patch_size: int = Field(16, description="Size of image patches.")
    num_channels: int = Field(3, description="Number of input image channels.")
    # ... other vision-specific fields

# --- Multi-Modal Config --- #
class MultiModalModelConfig(BaseModelConfig):
    """Placeholder Config for Multi-Modal Models"""
    model_type: str = Field("multimodal", description="Model type set to multimodal.")
    # References to language and vision configs, or combined fields
    language_config: Optional[LanguageModelConfig] = None 
    vision_config: Optional[VisionModelConfig] = None
    # ... other multimodal-specific fields 