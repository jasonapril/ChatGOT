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
    return {
        '_target_': 'tests.helpers.mock_components.MockModel',
        'config': { # Add nested config block required by initialize_model
            'architecture': 'mock_arch'
            # Add other minimal required params if MockModel expects them
        }
    }

@pytest.fixture
def minimal_experiment_config_node(minimal_training_config, minimal_model_config_dict):
    conf_dict = {
        'name': 'minimal_resume_test',
        'training': OmegaConf.create(minimal_training_config.model_dump()),
        'model': OmegaConf.create(minimal_model_config_dict),
        'data': OmegaConf.create({
            'datasets': {
                'train': { # Provide minimal valid config for train split
                    'dataset': {
                        '_target_': 'tests.helpers.mock_components.MockDataset',
                        'length': 10 # Example length
                    }
                 },
                'val': None,
                'test': None
            },
            'batch_size': 2 # Example batch size for dataloader wrapper
        }),
        'optimizer': {
            '_target_': 'torch.optim.AdamW',
            'lr': 1e-4
        },
        'checkpointing': {
            'dummy_key': True
        },
        'callbacks': None,
        'output_dir': None,
        'device': 'cpu'
    }
    return OmegaConf.create({'experiment': conf_dict})

class TestTrainerResume:
    """Tests for Trainer checkpoint resuming logic."""

    @pytest.fixture
    def trainer_instance_for_success(self, minimal_experiment_config_node, tmp_path):
        """Fixture for successful resume test, passes resume path to Trainer init."""
        resume_path = "/path/to/valid/checkpoint"
        loaded_state_obj = TrainingState(
            epoch=5, global_step=1000, best_val_metric=0.5,
            metrics={'loss': 0.6}, model_state_dict={}, optimizer_state_dict={}
        )

        # Patch initialize_cm WITHIN trainer module
        with patch('craft.training.trainer.initialize_checkpoint_manager') as mock_init_cm, \
             patch('craft.training.trainer.logging.getLogger') as mock_log, \
             patch('craft.training.initialization.initialize_device'), \
             patch('craft.training.initialization.initialize_tokenizer'), \
             patch('craft.training.initialization.initialize_model'), \
             patch('craft.training.initialization.initialize_dataloaders'), \
             patch('craft.training.initialization.initialize_optimizer'), \
             patch('craft.training.initialization.initialize_scheduler'), \
             patch('craft.training.initialization.initialize_callbacks'), \
             patch('craft.training.initialization.initialize_evaluator'), \
             patch('craft.training.initialization.initialize_amp_scaler'), \
             patch('craft.training.initialization.compile_model_if_enabled'):

            mock_logger = mock_log.return_value
            mock_cm_instance = MagicMock(spec=CheckpointManager)
            mock_init_cm.return_value = mock_cm_instance
            mock_cm_instance.load_checkpoint.return_value = loaded_state_obj
            mock_cm_instance.checkpoint_dir = tmp_path / "checkpoints"

            # Instantiate Trainer, explicitly passing resume path
            trainer = Trainer(
                cfg=minimal_experiment_config_node,
                resume_from_checkpoint=resume_path # Pass directly to init
            )

            # Assign logger for test assertions
            trainer.logger = mock_logger

            # Check the mock was assigned (optional sanity check)
            # assert trainer.checkpoint_manager is mock_cm_instance

            return trainer, loaded_state_obj, resume_path # Return path for assertion

    def test_resume_successful(self, trainer_instance_for_success):
        """Test successful resumption triggered during Trainer init."""
        trainer, loaded_state_obj, resume_path = trainer_instance_for_success

        # --- Assertions (Resume already happened in fixture's Trainer init) --- #
        # Check checkpoint manager's load_checkpoint was called correctly with positional arg
        trainer.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)

        # Check trainer state was updated
        assert trainer.epoch == loaded_state_obj.epoch
        assert trainer.global_step == loaded_state_obj.global_step
        assert trainer.best_val_metric == loaded_state_obj.best_val_metric

        # Check log message
        # Note: The exact log message might change, adjust if needed
        trainer.logger.info.assert_any_call(f"Resuming from Epoch: {loaded_state_obj.epoch}, Global Step: {loaded_state_obj.global_step}, Best Val Metric: {loaded_state_obj.best_val_metric}")
        trainer.logger.info.assert_any_call(f"Training resumed successfully from step {loaded_state_obj.global_step}.")

    # Separate test for failure, potentially without full fixture if needed
    def test_resume_failure(self, minimal_experiment_config_node, tmp_path):
        """Test failure during manual call to _resume_from_checkpoint."""
        resume_path = "/path/to/bad/checkpoint"
        error_message = "Simulated load failure"

        # Patch initialize_cm within trainer module for this test scope
        with patch('craft.training.trainer.initialize_checkpoint_manager') as mock_init_cm, \
             patch('craft.training.trainer.logging.getLogger') as mock_log, \
             patch('craft.training.initialization.initialize_device'), \
             patch('craft.training.initialization.initialize_tokenizer'), \
             patch('craft.training.initialization.initialize_model'), \
             patch('craft.training.initialization.initialize_dataloaders'), \
             patch('craft.training.initialization.initialize_optimizer'), \
             patch('craft.training.initialization.initialize_scheduler'), \
             patch('craft.training.initialization.initialize_callbacks'), \
             patch('craft.training.initialization.initialize_evaluator'), \
             patch('craft.training.initialization.initialize_amp_scaler'), \
             patch('craft.training.initialization.compile_model_if_enabled'):

            mock_logger = mock_log.return_value
            mock_cm_instance = MagicMock(spec=CheckpointManager)
            mock_init_cm.return_value = mock_cm_instance
            mock_cm_instance.load_checkpoint.side_effect = CheckpointLoadError(error_message)
            mock_cm_instance.checkpoint_dir = tmp_path / "checkpoints"

            # Instantiate Trainer WITHOUT passing resume path to init
            trainer = Trainer(cfg=minimal_experiment_config_node)
            trainer.logger = mock_logger # Assign mock logger
            # Manually assign the mock CM instance for this test
            # since init didn't trigger resume/CM usage here
            trainer.checkpoint_manager = mock_cm_instance

            # --- Action & Assertions --- #
            # Assert that *manually calling* _resume_from_checkpoint returns None and logs error
            # The method handles the exception internally and returns None
            return_value = trainer._resume_from_checkpoint(resume_path)
            assert return_value is None

            # Assert CheckpointManager's load was called and logger error
            trainer.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)
            trainer.logger.error.assert_called_once()
            assert error_message in trainer.logger.error.call_args[0][0] 