import pytest
import torch
from unittest.mock import MagicMock, patch
import logging

from craft.training.trainer import Trainer
from craft.training.checkpointing import CheckpointManager # Import needed for mock
from craft.config.schemas import TrainingConfig # Import TrainingConfig

class TestTrainerResume:
    """Tests for Trainer checkpoint resuming logic."""

    @pytest.fixture
    def trainer_instance(self, mock_model, mock_dataloader, mock_optimizer):
        """Fixture to create a Trainer instance with minimal mocks for resume testing."""
        with patch('craft.training.trainer.logging.getLogger'), \
             patch('craft.training.trainer.CheckpointManager') as mock_cm, \
             patch('craft.training.trainer.torch.amp.GradScaler'), \
             patch('craft.training.trainer.torch.device') as mock_dev:

            mock_dev.return_value = torch.device("cpu")
            # Prevent actual resume call during initialization for these tests
            # Create a minimal valid config
            minimal_config = TrainingConfig(batch_size=4)
            trainer = Trainer(
                model=mock_model,
                train_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                config=minimal_config # Pass the config object
                # resume_from_checkpoint=None # REMOVED: Handled by config
            )
            # Manually assign the mocked checkpoint manager AFTER init
            trainer.checkpoint_manager = mock_cm.return_value
            return trainer

    def test_resume_successful(self, trainer_instance, tmp_path):
        """Test successful resumption from a checkpoint."""
        # --- Setup --- #
        resume_path = "/path/to/checkpoint"
        state = {
            'epoch': 5,
            'global_step': 1000,
            'best_val_metric': 0.5,
            'metrics': {'loss': 0.6}
        }
        trainer_instance.checkpoint_manager.load_checkpoint.return_value = state
        trainer_instance.resume_from_checkpoint = resume_path # Set path to trigger resume
        trainer_instance.logger = MagicMock() # Mock logger for assertion

        # --- Action --- #
        trainer_instance._resume_from_checkpoint()

        # --- Assertions --- #
        trainer_instance.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)
        assert trainer_instance.epoch == state['epoch']
        assert trainer_instance.global_step == state['global_step']
        assert trainer_instance.best_val_metric == state['best_val_metric']
        assert trainer_instance.metrics == state['metrics']
        trainer_instance.logger.info.assert_called_once_with(
            f"Resumed trainer state from checkpoint at epoch {state['epoch']}, step {state['global_step']}"
        )

    def test_resume_failure(self, trainer_instance):
        """Test failure during checkpoint loading."""
        # --- Setup --- #
        resume_path = "/path/to/bad/checkpoint"
        error_message = "File not found"
        trainer_instance.checkpoint_manager.load_checkpoint.side_effect = FileNotFoundError(error_message)
        trainer_instance.resume_from_checkpoint = resume_path
        trainer_instance.logger = MagicMock()

        # --- Action & Assertions --- #
        with pytest.raises(FileNotFoundError, match=error_message):
            trainer_instance._resume_from_checkpoint()

        trainer_instance.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)
        trainer_instance.logger.error.assert_called_once()
        # Check that the error message logged contains the exception string
        assert error_message in trainer_instance.logger.error.call_args[0][0] 