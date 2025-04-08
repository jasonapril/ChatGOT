import pytest
import torch
from unittest.mock import MagicMock, patch
import logging
from omegaconf import OmegaConf # Add OmegaConf import
import torch.nn as nn

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
        'name': 'minimal_resume_test',
        'training': OmegaConf.create(minimal_training_config.model_dump()),
        'model': OmegaConf.create(minimal_model_config_dict),
        'data': OmegaConf.create({'datasets': {'train': None, 'val': None, 'test': None}}),
        'callbacks': None
    }
    return OmegaConf.create(conf_dict)

class TestTrainerResume:
    """Tests for Trainer checkpoint resuming logic."""

    @pytest.fixture
    def trainer_instance(self, minimal_training_config, minimal_model_config_dict, minimal_experiment_config_node, tmp_path):
        """Fixture to create a Trainer instance with minimal mocks for resume testing."""
        # --- Resume Setup --- #
        resume_path = "/path/to/checkpoint" # Define resume path
        loaded_state_obj = TrainingState( # Define loaded state object
            epoch=5,
            global_step=1000,
            best_val_metric=0.5,
            metrics={'loss': 0.6},
            model_state_dict={},
            optimizer_state_dict={}
        )
        # Modify config to include resume path *before* Trainer init
        config_with_resume = minimal_training_config.model_copy(update={"resume_from_checkpoint": resume_path})

        # --- Mocks --- #
        mock_model = MagicMock(spec=nn.Module)
        mock_optimizer = MagicMock(spec=torch.optim.Optimizer)
        mock_dataloader = MagicMock(spec=torch.utils.data.DataLoader)

        with patch('craft.training.trainer.logging.getLogger') as mock_log, \
             patch('craft.training.trainer.CheckpointManager') as mock_cm, \
             patch('craft.training.trainer.torch.amp.GradScaler'):

            mock_logger = mock_log.return_value
            mock_cm_instance = mock_cm.return_value # Get the instance from the patch

            # Configure mock CheckpointManager *before* Trainer init
            mock_cm_instance.load_checkpoint.return_value = loaded_state_obj
            mock_cm_instance.checkpoint_dir = tmp_path / "checkpoints"

            # --- Mock component state loading methods BEFORE Trainer init --- #
            mock_model.load_state_dict = MagicMock()
            mock_optimizer.load_state_dict = MagicMock()
            # Mock scheduler/scaler if they are created/passed and loaded

            # --- Instantiate Trainer (should trigger resume) --- #
            trainer = Trainer(
                config=config_with_resume, # Pass config with resume path
                model=mock_model, # Pass mock
                optimizer=mock_optimizer, # Pass mock
                train_dataloader=mock_dataloader, # Pass mock
                experiment_name=minimal_experiment_config_node.name, # Keep experiment_name if needed
                device="cpu",
                checkpoint_manager=mock_cm_instance, # Pass pre-configured mock manager
            )

            # --- Post-Init Setup --- #
            # Mock components that might be accessed *after* init/resume for assertions
            trainer.logger = mock_logger # Assign logger for tests
            # Mock components needed for potential state application validation (if done after init)
            # Ensure these mocks exist on the trainer *after* initialization
            trainer.model = mock_model # Re-assign if Trainer replaces it? Probably not needed if passed in init
            trainer.optimizer = mock_optimizer
            # trainer.scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler) # Add if needed
            # trainer.scaler = MagicMock(spec=torch.cuda.amp.GradScaler) # Add if needed

            # Return the trainer and the expected loaded state for assertions
            return trainer, loaded_state_obj

    def test_resume_successful(self, trainer_instance):
        """Test successful resumption from a checkpoint (triggered during init)."""
        # --- Setup (Now mostly done in fixture) --- #
        trainer, loaded_state_obj = trainer_instance # Unpack trainer and expected state
        resume_path = trainer.config.resume_from_checkpoint # Get resume path from trainer's config

        # --- Action (Resume happens during Trainer.__init__) --- #
        # No action needed here

        # --- Assertions --- #
        # Check checkpoint manager was called correctly during init
        trainer.checkpoint_manager.load_checkpoint.assert_called_once_with(path_specifier=resume_path)
        # Check that trainer attributes were updated correctly during init/resume
        assert trainer.epoch == loaded_state_obj.epoch
        assert trainer.global_step == loaded_state_obj.global_step
        assert trainer.best_val_metric == loaded_state_obj.best_val_metric
        # Assert log message (ensure logger is correctly mocked/assigned in fixture)
        trainer.logger.info.assert_any_call(
            f"Successfully resumed trainer state to Step={loaded_state_obj.global_step}, Epoch={loaded_state_obj.epoch}, BestMetric={loaded_state_obj.best_val_metric}"
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