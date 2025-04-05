"""
Unit tests for the TransformerModel class.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from craft.models.configs import LanguageModelConfig # Transformer uses LM config
from craft.models.transformer import TransformerModel

class TestTransformerModel:
    """Tests for the TransformerModel class (pytest style)."""
    
    @pytest.fixture
    def transformer_config(self):
        """Provides a standard TransformerModelConfig (using LanguageModelConfig)."""
        # TransformerModel expects a LanguageModelConfig
        return LanguageModelConfig(
            model_type="language", # Needs to be set, even if arch specific
            model_architecture="transformer", # Set arch explicitly
            vocab_size=100, 
            d_model=64, 
            n_head=4, 
            n_layers=2,
            max_seq_length=512 # Added max_seq_length as it's required
            # Other fields like dropout, bias use defaults
        )

    @pytest.fixture
    def transformer_model(self, transformer_config):
        """Provides a TransformerModel instance."""
        return TransformerModel(config=transformer_config)

    def test_forward_pass_shape(self, transformer_model):
        """Test the output shape of the forward pass."""
        input_ids = torch.randint(0, transformer_model.config.vocab_size, (5, 10))
        output = transformer_model(input_ids)
        
        # Expected shape: (batch_size, seq_len, vocab_size)
        assert output.shape == (5, 10, transformer_model.config.vocab_size) 