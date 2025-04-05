"""
Unit tests for the LanguageModel base class.
"""
import pytest
import torch

from craft.models.base import Model, GenerativeModel, LanguageModel
from craft.models.configs import LanguageModelConfig
from .conftest import MockLanguageModel

class TestLanguageModel:
    """Tests for the LanguageModel class (using pytest style)."""
    
    @pytest.fixture
    def language_model(self):
        """Fixture for MockLanguageModel."""
        return MockLanguageModel()
    
    def test_initialization(self, language_model):
        """Test that the model initializes correctly."""
        assert isinstance(language_model.config, LanguageModelConfig)
        assert language_model.model_type == "language"
        # Check inheritance
        assert isinstance(language_model, GenerativeModel)
        assert isinstance(language_model, Model)
    
    def test_forward(self, language_model):
        """Test the forward method."""
        x = torch.randint(0, 100, (5, 10))
        output = language_model(x)
        assert output.shape == (5, 10, 100)
    
    def test_generate_method(self, language_model):
        """Test the generate method."""
        input_ids = torch.randint(0, 100, (2, 5))
        output = language_model.generate(input_ids, max_new_tokens=15)
        assert output.shape == (2, 20)
    
    def test_calculate_perplexity(self, language_model):
        """Test the calculate_perplexity method."""
        logits = torch.randn(2, 5, 100)
        targets = torch.randint(0, 100, (2, 5))
        perplexity = language_model.calculate_perplexity(logits, targets)
        assert perplexity.item() > 0 