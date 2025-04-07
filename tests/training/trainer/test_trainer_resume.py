import pytest
import torch
from unittest.mock import MagicMock, patch
import logging
from omegaconf import OmegaConf # Add OmegaConf import

from craft.training.trainer import Trainer
from craft.training.checkpointing import CheckpointManager, CheckpointLoadError, TrainingState # Import needed for mock & state
from craft.config.schemas import TrainingConfig # Import TrainingConfig

# Minimal Config Fixtures (copied from other test files)
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

class TestTrainerResume:
    """Tests for Trainer checkpoint resuming logic."""

    @pytest.fixture
    def trainer_instance(self, minimal_training_config, minimal_model_config_dict, minimal_experiment_config_node):
        """Fixture to create a Trainer instance with minimal mocks for resume testing."""
        with patch('craft.training.trainer.logging.getLogger') as mock_log, \
             patch('craft.training.trainer.CheckpointManager') as mock_cm, \
             patch('craft.training.trainer.torch.amp.GradScaler'), \
             patch('craft.training.trainer.torch.device') as mock_dev:

            mock_dev.return_value = torch.device("cpu")
            mock_logger = mock_log.return_value
            mock_cm_instance = mock_cm.return_value # Get the instance from the patch

            # --- Instantiate Trainer with new signature --- #
            trainer = Trainer(
                model_config=minimal_model_config_dict,
                config=minimal_training_config,
                experiment_config=minimal_experiment_config_node,
                device="cpu",
                experiment_name="test_resume_init"
                # No resume path passed here, tests will set it manually or rely on internal call
            )

            # Ensure the trainer uses the mocked CM instance created by __init__ from the patched class
            trainer.checkpoint_manager = mock_cm_instance
            trainer.logger = mock_logger # Assign logger for tests
            # Manually initialize state for resume tests
            trainer.state = TrainingState() 

            return trainer

    def test_resume_successful(self, trainer_instance, tmp_path):
        """Test successful resumption from a checkpoint."""
        # --- Setup --- #
        resume_path = "/path/to/checkpoint"
        # Mock load_checkpoint to return a TrainingState object, not a dict
        loaded_state_obj = TrainingState(
            epoch=5,
            global_step=1000,
            best_val_metric=0.5,
            metrics={'loss': 0.6},
            model_state_dict={}, # Add dummy required field
            optimizer_state_dict={} # Add dummy required field
        )
        trainer_instance.checkpoint_manager.load_checkpoint.return_value = loaded_state_obj
        trainer_instance.resume_from_checkpoint = resume_path # Set path to trigger resume
        trainer_instance.logger = MagicMock() # Mock logger for assertion

        # --- Action --- #
        trainer_instance._resume_from_checkpoint(resume_path)
    
        # --- Assertions --- #
        trainer_instance.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)
        # Check that trainer attributes are updated
        assert trainer_instance.state.epoch == loaded_state_obj.epoch
        assert trainer_instance.state.global_step == loaded_state_obj.global_step
        assert trainer_instance.state.best_val_metric == loaded_state_obj.best_val_metric
        assert trainer_instance.state.metrics == loaded_state_obj.metrics
        # Check for one of the actual log messages
        trainer_instance.logger.info.assert_any_call(
            f"Resumed from Epoch {loaded_state_obj.epoch}, Global Step {loaded_state_obj.global_step}"
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
            # Pass the checkpoint path argument
            trainer_instance._resume_from_checkpoint(resume_path)

        trainer_instance.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)
        trainer_instance.logger.error.assert_called_once()
        # Check that the error message logged contains the exception string
        assert error_message in trainer_instance.logger.error.call_args[0][0] 