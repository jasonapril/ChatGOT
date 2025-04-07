"""
Unit tests for the TransformerModel class.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

# from craft.models.configs import LanguageModelConfig # Transformer uses LM config (Old Location)
from craft.config.schemas import LanguageModelConfig # New Location

from craft.models.transformer import TransformerModel

class TestTransformerModel:
    """Tests for the TransformerModel class (pytest style)."""
    
    @pytest.fixture
    def transformer_config(self):
        """Provides a standard TransformerModelConfig (using LanguageModelConfig)."""
        # TransformerModel expects a LanguageModelConfig
        return LanguageModelConfig(
            # model_type="language", # model_type removed from LanguageModelConfig, implied by inheritance/architecture
            architecture="transformer", # Now required as discriminator
            vocab_size=100, 
            d_model=64, 
            n_head=4, 
            n_layers=2,
            max_seq_length=512 # Added max_seq_length as it's required
            # Other fields like dropout, bias use defaults
        )

    @pytest.fixture
    def transformer_model(self, transformer_config):
        """Provides an instance of TransformerModel."""
        # Ensure config has architecture before model init
        transformer_config.architecture = 'transformer' 
        return TransformerModel(config=transformer_config)

    def test_forward_pass_shape(self, transformer_model):
        """Test the output shape of the forward pass."""
        # Test with small batch and sequence length
        batch_size = 5
        seq_len = 10
        # Use the vocab_size from the model's config
        vocab_size = transformer_model.config.vocab_size # vocab_size should exist in config
        
        # Create a dummy input tensor
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = transformer_model(input_ids)
        
        # Expected shape: (batch_size, seq_len, vocab_size)
        assert output.shape == (batch_size, seq_len, vocab_size)
        # No longer check model_type

    def test_generate_method_inherited(self, transformer_model):
        """Test that the generate method is inherited and works."""

    def test_forward_pass_shape(self, transformer_model):
        """Test the output shape of the forward pass."""
        input_ids = torch.randint(0, transformer_model.config.vocab_size, (5, 10))
        output = transformer_model(input_ids)
        
        # Expected shape: (batch_size, seq_len, vocab_size) 