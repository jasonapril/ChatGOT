import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
import logging
from omegaconf import OmegaConf

# Import the class to test
from craft.training.trainer import Trainer
from craft.config.schemas import TrainingConfig # Import TrainingConfig

# Minimal Config Fixtures (similar to test_trainer_init.py)
@pytest.fixture
def minimal_training_config():
    return TrainingConfig(batch_size=2, num_epochs=1)

@pytest.fixture
def minimal_model_config_dict():
    return {'architecture': 'mock_arch', '_target_': 'tests.conftest.MockModel'}

@pytest.fixture
def minimal_experiment_config_node(minimal_training_config, minimal_model_config_dict):
    conf_dict = {
        'training': OmegaConf.create(minimal_training_config.model_dump()),
        'model': OmegaConf.create(minimal_model_config_dict),
        'data': OmegaConf.create({'datasets': {'train': None, 'val': None, 'test': None}}),
        'callbacks': None
    }
    return OmegaConf.create(conf_dict)

@patch('craft.training.trainer.TextGenerator')
class TestTrainerGenerate:
    """Tests for the Trainer.generate_text method."""

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    @patch('hydra.utils.instantiate') # Patch hydra
    def test_generate_text_passes_args(self, 
                                       mock_hydra_instantiate, # Add arg
                                       mock_torch_device, 
                                       mock_grad_scaler, 
                                       mock_checkpoint_manager, 
                                       mock_get_logger, 
                                       mock_text_generator, 
                                       mock_model, # Keep model mock for generate
                                       mock_tokenizer, # Add tokenizer mock
                                       minimal_training_config, # Add config fixtures
                                       minimal_model_config_dict, 
                                       minimal_experiment_config_node):
        """Test that generate_text initializes TextGenerator and passes args."""
        # Configure hydra mock
        mock_hydra_instantiate.return_value = {} # Return empty dict

        # --- Setup --- #
        # Mock device
        mock_cpu_device = torch.device("cpu")
        mock_torch_device.return_value = mock_cpu_device

        # Create a minimal trainer instance with the mock config
        trainer = Trainer(
            model_config=minimal_model_config_dict,
            config=minimal_training_config,
            experiment_config=minimal_experiment_config_node,
            device="cpu",
            experiment_name="test_generate"
        )
        
        # Manually set model and tokenizer as setup() is not called
        trainer._model = mock_model
        trainer._tokenizer = mock_tokenizer

        # Mock the generator instance returned by the __init__ of TextGenerator
        mock_generator_instance = mock_text_generator.return_value
        mock_generator_instance.generate_text.return_value = ["Generated text"]

        prompt = "Test prompt"
        max_new = 50
        temp = 0.7
        
        # --- Call Method ---
        result = trainer.generate_text(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temp,
            # Pass other args if needed for testing
        )

        # --- Assertions ---
        # Check TextGenerator was initialized correctly
        mock_text_generator.assert_called_once_with(
            model=trainer._model, # Should use the manually set model
            tokenizer=trainer._tokenizer, # Should use the manually set tokenizer
            device=mock_cpu_device,
            # config=minimal_training_config # Remove config argument, likely not needed
        )

        # Check the generate_text method on the *instance* was called
        mock_generator_instance.generate_text.assert_called_once_with(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temp,
            # Add other expected default args based on TextGenerator defaults
            top_k=40,          # Default from TextGenerator
            top_p=0.9,         # Default from TextGenerator
            repetition_penalty=1.0, # Default from TextGenerator
            num_return_sequences=1, # Default from Trainer.generate_text
            use_beam_search=False, # Default from TextGenerator
            num_beams=5,           # Default from TextGenerator
            length_penalty=1.0,    # Default from TextGenerator
            early_stopping=True    # Default from TextGenerator
        )

        assert result == ["Generated text"] 