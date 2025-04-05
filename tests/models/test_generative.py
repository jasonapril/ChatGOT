"""
Unit tests for the GenerativeModel base class.
"""
import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError
from craft.models.base import Model, GenerativeModel
from craft.models.configs import GenerativeModelConfig
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