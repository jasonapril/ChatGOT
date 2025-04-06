import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
import logging

# Import the class to test
from craft.training.trainer import Trainer
from craft.config.schemas import TrainingConfig # Import TrainingConfig

@patch('craft.training.trainer.TextGenerator')
class TestTrainerGenerate:
    """Tests for the Trainer.generate_text method."""

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    def test_generate_text_passes_args(self, mock_torch_device, mock_grad_scaler, mock_checkpoint_manager, mock_get_logger, mock_text_generator, mock_model, mock_dataloader, mock_optimizer):
        """Test that generate_text initializes TextGenerator and passes args."""
        # --- Setup --- #
        # Mock device
        mock_cpu_device = torch.device("cpu")
        mock_torch_device.return_value = mock_cpu_device

        # Mock TrainingConfig and its model_dump
        mock_config_obj = MagicMock(spec=TrainingConfig)
        # Mock attributes accessed by Trainer.__init__
        mock_config_obj.num_epochs = 1
        mock_config_obj.max_steps = None
        mock_config_obj.use_amp = False
        mock_config_obj.gradient_accumulation_steps = 1
        mock_config_obj.max_grad_norm = None
        mock_config_obj.log_interval = 10
        mock_config_obj.eval_interval = 100
        mock_config_obj.save_interval = 500
        mock_config_obj.time_save_interval_seconds = None
        mock_config_obj.time_eval_interval_seconds = None
        mock_config_obj.save_steps_interval = 0
        mock_config_obj.checkpoint_dir = None
        mock_config_obj.keep_last = None
        mock_config_obj.batch_size = 32 # Need this for init
        # Mock the model_dump method used by CheckpointManager
        mock_config_dict = {'batch_size': 32} # Example dumped dict
        mock_config_obj.model_dump.return_value = mock_config_dict
        
        # Create a minimal trainer instance with the mock config
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=mock_config_obj # Pass the mock TrainingConfig object
        )
        
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
            model=mock_model,
            device=mock_cpu_device,
            config=mock_config_obj # Add config to expected call
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