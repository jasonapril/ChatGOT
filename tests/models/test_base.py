"""
Unit tests for the base Model class.
"""
import pytest
import torch
import torch.nn as nn

from craft.models.base import Model, BaseModelConfig
from .conftest import ConcreteModel # Import concrete implementation for testing

# Tests for the Model base class (nn.Module)
class TestBaseModel:
    """Tests for the Model base class using pytest fixtures."""

    @pytest.fixture
    def config(self):
        """Provides a basic BaseModelConfig fixture."""
        return BaseModelConfig(model_type="test_base")

    @pytest.fixture
    def concrete_model(self, config):
        """Provides an instance of the ConcreteModel fixture."""
        return ConcreteModel(config)

    def test_initialization(self, concrete_model, config):
        """Test correct initialization."""
        assert concrete_model.config == config
        assert concrete_model.model_type == "test_base"
        assert isinstance(concrete_model.linear, nn.Linear)
        config['architecture'] = 'mock_arch'

    def test_get_config(self, concrete_model, config):
        """Test the get_config method."""
        retrieved_config = concrete_model.get_config()
        assert retrieved_config == config.model_dump()
        config['architecture'] = 'mock_arch'

    def test_forward(self, concrete_model):
        """Test the forward method (concrete implementation)."""
        # The abstract Model.forward raises NotImplementedError.
        # We test the concrete implementation here.
        input_tensor = torch.randn(1, 10)
        output = concrete_model.forward(input_tensor)
        assert output.shape == (1, 5) # Based on ConcreteModel's linear layer
        concrete_model.config['architecture'] = 'mock_arch'

    def test_save_load(self, concrete_model, tmp_path):
        """Test the save and load methods."""
        save_path = tmp_path / "model.pt"
        extra_data = {"test_key": "test_value"}
        
        # Save the original model (likely on CPU)
        original_device = next(concrete_model.parameters()).device
        concrete_model.save(str(save_path), **extra_data)
        assert save_path.exists()

        # Create a new model instance of the same type to load into
        new_model = ConcreteModel(concrete_model.config)

        # Ensure parameters are different before loading (comparing on original device)
        assert not torch.equal(concrete_model.linear.weight.to(original_device), 
                               new_model.linear.weight.to(original_device))

        # Load the saved state into the new model.
        # The load method will move new_model to the detected device (CPU or CUDA)
        loaded_data = new_model.load(str(save_path))
        loaded_device = next(new_model.parameters()).device

        # Move the original model to the same device as the loaded model for comparison
        concrete_model.to(loaded_device)

        # Check if model state is loaded correctly (now comparing on the same device)
        assert torch.equal(concrete_model.linear.weight, new_model.linear.weight)
        assert torch.equal(concrete_model.linear.bias, new_model.linear.bias)

        # Check config compatibility (implicitly tested by successful load)
        # Verify that the extra data was returned from load
        assert "test_key" in loaded_data 
        concrete_model.config['architecture'] = 'mock_arch'
        extra_data['architecture'] = 'mock_arch'
        assert loaded_data == extra_data 