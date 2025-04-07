"""
Unit tests for the LanguageModel base class.
"""
import pytest
import torch

from craft.models.base import Model, GenerativeModel, LanguageModel
from craft.config.schemas import LanguageModelConfig
from .conftest import MockLanguageModel

class TestLanguageModel:
    """Tests for the LanguageModel class (using pytest style)."""
    
    @pytest.fixture
    def language_model(self):
        """Fixture for MockLanguageModel."""
        return MockLanguageModel()
    
    def test_initialization(self, mock_language_config_dict):
        """Test that the model initializes correctly."""
        # Add discriminator
        mock_language_config_dict['architecture'] = 'mock_language_arch'
        model = MockLanguageModel(mock_language_config_dict)
        assert isinstance(model.config, LanguageModelConfig)
        assert model.config.d_model == mock_language_config_dict['d_model']
        # Check other inherited fields if necessary (e.g., max_seq_length)
        assert hasattr(model.config, 'max_seq_length')
    
    def test_forward(self, mock_language_config_dict):
        """Test the forward method."""
        # Add discriminator
        mock_language_config_dict['architecture'] = 'mock_language_arch'
        model = MockLanguageModel(mock_language_config_dict)
        batch_size = 2
        seq_len = 10
        x = torch.randint(0, 100, (batch_size, seq_len))
        output = model(x)
        assert output.shape == (batch_size, seq_len, 100)
    
    def test_generate_method(self, mock_language_config_dict):
        """Test the generate method."""
        # Add discriminator
        mock_language_config_dict['architecture'] = 'mock_language_arch'
        model = MockLanguageModel(mock_language_config_dict)
        # Test if generate method exists (inherited from GenerativeModel)
        assert hasattr(model, 'generate')
        input_ids = torch.randint(0, 100, (2, 5))
        output = model.generate(input_ids, max_new_tokens=15)
        assert output.shape == (2, 20)
    
    def test_calculate_perplexity(self, mock_language_config_dict):
        """Test the calculate_perplexity method."""
        # Add discriminator
        mock_language_config_dict['architecture'] = 'mock_language_arch'
        model = MockLanguageModel(mock_language_config_dict)
        # Prepare dummy data
        dummy_logits = torch.randn(2, 5, model.config.vocab_size) # (batch, seq, vocab)
        targets = torch.randint(0, 100, (2, 5))
        perplexity = model.calculate_perplexity(dummy_logits, targets)
        assert perplexity.item() > 0 