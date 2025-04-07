"""
Unit tests for the GenerativeModel base class.
"""
import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError
from craft.models.base import Model, GenerativeModel
from craft.config.schemas import GenerativeModelConfig
from .conftest import MockGenerativeModel

class TestGenerativeModel:
    """Tests for the GenerativeModel class (using pytest style)."""
    
    @pytest.fixture
    def generative_model(self):
        """Fixture for MockGenerativeModel."""
        return MockGenerativeModel()
    
    def test_initialization(self, generative_model):
        """Test that the model initializes correctly."""
        assert isinstance(generative_model.config, GenerativeModelConfig)
        assert generative_model.model_type == "generative"
        # Check that it IS an instance of the base Model class
        assert isinstance(generative_model, Model)
        generative_model.config.architecture = 'mock_generative_arch'
        assert generative_model.config.max_seq_length == generative_model.config.max_seq_length

    def test_initialization_with_config(self, mock_generative_config_dict):
        mock_generative_config_dict['architecture'] = 'mock_generative_arch'
        model = MockGenerativeModel(mock_generative_config_dict)
        assert isinstance(model.config, GenerativeModelConfig)
        assert model.config.max_seq_length == mock_generative_config_dict['max_seq_length'] 