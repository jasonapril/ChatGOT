"""
Configuration and fixtures for model tests.
"""
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from unittest.mock import MagicMock

# Import base model classes
from craft.models.base import Model, GenerativeModel, LanguageModel, VisionModel, MultiModalModel
# Import config classes from the correct location
from craft.models.configs import (
    BaseModelConfig, # Import the actual base config
    GenerativeModelConfig,
    LanguageModelConfig,
    VisionModelConfig,
    MultiModalModelConfig
)

# --- Mock Classes --- #

# Inherit from Model (nn.Module base), NOT pydantic.BaseModel
class MockBaseModel(Model):
    """Mock implementation of BaseModel for testing."""
    def __init__(self):
        super().__init__(config=BaseModelConfig())
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

# Inherit from GenerativeModel (which inherits Model)
class MockGenerativeModel(GenerativeModel):
    """Mock implementation of GenerativeModel for testing generate method."""
    def __init__(self, vocab_size=10, d_model=8, max_seq_length=20):
        # Use a config that matches expected attributes
        config = GenerativeModelConfig(
            model_type="generative", 
            max_seq_length=max_seq_length
        )
        # We need d_model for the dummy linear layer
        config.d_model = d_model # Add dynamically since it's not in GenerativeModelConfig
        config.vocab_size = vocab_size # Add dynamically for dummy output layer
        
        super().__init__(config=config)
        # Simple linear layer to mimic output projection
        self.linear = nn.Linear(d_model, vocab_size) 
    
    # This forward method will be mocked during tests
    def forward(self, x: torch.Tensor):
        # Dummy forward: creates output features based on input indices
        # The actual logits will be controlled by the mock object in tests.
        # Needs an embedding-like step for dimensionality change.
        # Let's simulate a simple projection based on input shape.
        batch_size, seq_len = x.shape
        # Dummy projection to d_model
        dummy_features = torch.randn(batch_size, seq_len, self.config.d_model, device=x.device)
        # Project to vocab size
        logits = self.linear(dummy_features)
        return logits

# Inherit from LanguageModel (which inherits GenerativeModel)
class MockLanguageModel(LanguageModel):
    """Mock implementation of LanguageModel for testing."""
    def __init__(self):
        config = LanguageModelConfig(vocab_size=100)
        super().__init__(config=config)
        self.embedding = nn.Embedding(self.config.vocab_size, 10)
        self.linear = nn.Linear(10, self.config.vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)
    
    # Override generate for simple test
    def generate(self, input_ids, max_new_tokens=10):
        batch_size = input_ids.shape[0]
        return torch.zeros(batch_size, input_ids.shape[1] + max_new_tokens)

# Simple concrete model for testing base functionalities
class ConcreteModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(10, 5) # Example layer

    def forward(self, x):
        # Basic implementation for testing
        return self.linear(x)

# Note: Fixtures using these mocks will be defined in the specific test files 
# or potentially here if shared across multiple test files. 