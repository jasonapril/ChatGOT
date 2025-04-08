import pytest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

# Assuming fixtures (setup_minimal_trainer_env, setup_optional_trainer_env)
# and create_instantiate_side_effect are available from conftest.py
from craft.training.trainer import Trainer
from craft.config.schemas import TrainingConfig
from craft.training.callbacks.base import CallbackList, Callback # For mock_callback_list_instance
from torch.cuda.amp import GradScaler as CudaGradScaler # For mock_scaler

# Dummy class needed for callback test target within test_init_with_optionals
# (even though the test file containing the definition might change later)
class MockCallbackTarget:
    pass

# =================================
# Initialization Tests - Basic & Optionals
# =================================
def test_init_minimal_required(setup_minimal_trainer_env: dict):
    """Test Trainer initialization with the minimal required configuration using instantiate patch."""

    # Arrange
    env = setup_minimal_trainer_env
    cfg = env["experiment_config"]

    instantiate_side_effect = create_instantiate_side_effect(env)

    # RE-ADD mock_model_class definition
    mock_model_class = MagicMock(spec=type, return_value=env["mock_model"])
    mock_loader_instance = env["mock_train_loader"]

    # Patch DataLoader *in the trainer module*
    with (patch('craft.training.trainer.instantiate', side_effect=instantiate_side_effect) as mock_instantiate,
          patch('hydra.utils.get_class', return_value=mock_model_class) as mock_get_class,
          patch('craft.training.trainer.DataLoader', return_value=mock_loader_instance) as mock_dataloader_patch):

        # Act
        trainer = Trainer(cfg=cfg)

        # Assert: Check if internal attributes are set correctly
        assert trainer.train_dataloader is mock_loader_instance
        assert isinstance(trainer.config, TrainingConfig)
        assert trainer.config == env["expected_training_config"]
        assert str(trainer.device) == cfg.experiment.device

        # Assert correct mock instances are assigned
        assert trainer.tokenizer is env["mock_tokenizer"]
        assert trainer.model is env["mock_model"]
        assert trainer.train_dataloader.dataset is env["mock_train_dataset"]
        assert trainer.optimizer is env["mock_optimizer"]
        assert trainer.scheduler is None
        assert trainer.val_dataloader is None
        assert trainer.evaluator is None
        assert trainer.checkpoint_manager is None
        assert trainer.scaler is not None # scaler is always created, enabled based on config.use_amp

        # Assert patched calls
        mock_instantiate.assert_any_call(cfg.experiment.data.tokenizer)
        mock_instantiate.assert_any_call(cfg.experiment.data.datasets.train)
        mock_instantiate.assert_any_call(cfg.experiment.optimizer, params=env["mock_model"].parameters())
        # Check get_class for model
        mock_get_class.assert_called_once_with(cfg.experiment.model._target_)
        mock_model_class.assert_called_once() # Model constructor called
        # Check DataLoader patch call
        mock_dataloader_patch.assert_called_once_with(env["mock_train_dataset"])

        # Use torch.device in assertion
        env["mock_model"].to.assert_called_with(torch.device(trainer.device))


@patch('tests.training.trainer.test_trainer_init.MockCallbackTarget') # Patch original location
def test_init_with_optionals(
    MockCallbackTarget_p, # Keep callback patch
    setup_optional_trainer_env, # Use optional env
    # tmp_path # tmp_path might be needed if CheckpointManager uses it
):
    """Test Trainer initialization with optional components using instantiate patch."""
    # Arrange
    env = setup_optional_trainer_env
    cfg = env["experiment_config"]
    mock_callback = env["mock_callback"]
    mock_scaler = env["mock_scaler"]

    # Configure callback target patch
    MockCallbackTarget_p.return_value = mock_callback

    instantiate_side_effect = create_instantiate_side_effect(env)

    # RE-ADD mock_model_class definition
    mock_model_class = MagicMock(spec=type, return_value=env["mock_model"])
    mock_callback_list_instance = MagicMock(spec=CallbackList)
    mock_callback_list_instance.callbacks = [mock_callback]

    # --- Mock DataLoader initialization (using side_effect) --- #
    def dataloader_side_effect(dataset, **kwargs):
        if dataset is env["mock_train_dataset"]:
            return env["mock_train_loader"]
        elif dataset is env["mock_val_dataset"]:
            return env["mock_val_loader"]
        raise ValueError(f"Unexpected DataLoader dataset: {type(dataset)}")
    mock_dataloader_init = MagicMock(side_effect=dataloader_side_effect)

    with (patch('craft.training.trainer.instantiate', side_effect=instantiate_side_effect) as mock_instantiate,
            patch('hydra.utils.get_class', return_value=mock_model_class) as mock_get_class,
            patch('craft.training.trainer.DataLoader', mock_dataloader_init) as mock_dataloader_patch,
            patch("craft.training.trainer.CallbackList", return_value=mock_callback_list_instance) as mock_callbacklist_init,
            patch("torch.cuda.amp.GradScaler", return_value=mock_scaler) as mock_gradscaler_init):

        # Act
        try:
            trainer = Trainer(cfg=cfg)
        except Exception as e:
            pytest.fail(f"Trainer initialization with optionals failed unexpectedly: {e}")

        # Assert: Check internal attributes
        assert isinstance(trainer.config, TrainingConfig)
        assert trainer.config == env["expected_training_config"]
        assert str(trainer.device) == cfg.experiment.device

        # Assert correct mock instances assigned
        assert trainer.tokenizer is env["mock_tokenizer"]
        assert trainer.model is env["mock_model"]
        assert trainer.train_dataloader is env["mock_train_loader"]
        assert trainer.val_dataloader is env["mock_val_loader"]
        assert trainer.optimizer is env["mock_optimizer"]
        assert trainer.scheduler is env["mock_scheduler"]
        assert trainer.callbacks is mock_callback_list_instance
        assert trainer.evaluator is env["mock_evaluator"]
        assert trainer.checkpoint_manager is env["mock_checkpoint_manager"]
        assert trainer.scaler is mock_scaler # Since use_amp=True

        # Assert patched calls
        instantiate_targets_called = {call.args[0].get('_target_') for call in mock_instantiate.call_args_list if isinstance(call.args[0], (dict, DictConfig))}
        assert cfg.experiment.data.tokenizer._target_ in instantiate_targets_called
        assert cfg.experiment.data.datasets.train._target_ in instantiate_targets_called
        assert cfg.experiment.data.datasets.val._target_ in instantiate_targets_called
        assert cfg.experiment.optimizer._target_ in instantiate_targets_called # Optimizer instantiated
        assert cfg.experiment.scheduler._target_ in instantiate_targets_called # Scheduler instantiated
        # assert cfg.experiment.evaluation._target_ in instantiate_targets_called # Evaluator might be None if no val_loader
        # assert cfg.experiment.checkpointing._target_ in instantiate_targets_called # CkptManager might be None
        assert 'tests.training.trainer.test_trainer_init.MockCallbackTarget' in instantiate_targets_called # Callback target instantiated

        # Check direct calls
        mock_get_class.assert_called_once_with(cfg.experiment.model._target_)
        mock_model_class.assert_called_once()
        # Assert DataLoader patch calls
        mock_dataloader_patch.assert_any_call(env["mock_train_dataset"])
        mock_dataloader_patch.assert_any_call(env["mock_val_dataset"])
        # Check CallbackList init call (mocked instance)
        mock_callbacklist_init.assert_called_once_with([mock_callback])
        # Check GradScaler init
        mock_gradscaler_init.assert_called_once_with(enabled=True)

        # Use torch.device in assertion
        env["mock_model"].to.assert_called_with(torch.device(trainer.device)) 