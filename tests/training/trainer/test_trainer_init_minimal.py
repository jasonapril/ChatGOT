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
from craft.training.training_loop import TrainingLoop # Needed for final assertion

# =================================
# Minimal Initialization Test
# =================================
@patch("craft.training.trainer.initialize_device")
@patch("craft.training.trainer.initialize_tokenizer")
@patch("craft.training.trainer.initialize_model")
@patch("craft.training.trainer.initialize_dataloaders")
@patch("craft.training.trainer.initialize_optimizer")
@patch("craft.training.trainer.initialize_scheduler")
@patch("craft.training.trainer.initialize_amp_scaler")
@patch("craft.training.trainer.initialize_callbacks")
@patch("craft.training.trainer.initialize_checkpoint_manager")
@patch("craft.training.trainer.initialize_evaluator")
@patch("craft.training.trainer.compile_model_if_enabled")
@patch("craft.training.trainer.CallbackList") # Keep CallbackList patch
def test_init_minimal_required(
    # Mocks passed in reverse order
    MockCallbackList, # Keep
    mock_compile_model, # Keep
    mock_initialize_evaluator,
    mock_initialize_checkpoint_manager,
    mock_initialize_callbacks,
    mock_initialize_amp_scaler,
    mock_initialize_scheduler,
    mock_initialize_optimizer,
    mock_initialize_dataloaders,
    mock_initialize_model,
    mock_initialize_tokenizer,
    mock_initialize_device,
    setup_minimal_trainer_env: dict
):
    """Test Trainer initialization with the minimal required configuration using initialize patches."""

    # Arrange
    env = setup_minimal_trainer_env
    cfg = env["experiment_config"]

    # Configure the mocks to return values from the env fixture
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_tokenizer.return_value = env["mock_tokenizer"]
    mock_initialize_model.return_value = env["mock_model"]
    mock_initialize_dataloaders.return_value = (env["mock_train_loader"], env["mock_val_loader"]) # Tuple
    mock_initialize_optimizer.return_value = env["mock_optimizer"]
    mock_initialize_scheduler.return_value = env["mock_scheduler"]
    mock_initialize_amp_scaler.return_value = env["mock_scaler"]
    mock_initialize_callbacks.return_value = [] # No raw callbacks in minimal
    mock_initialize_checkpoint_manager.return_value = env["mock_checkpoint_manager"]
    mock_initialize_evaluator.return_value = env["mock_evaluator"]
    mock_compile_model.side_effect = lambda model, *a, **kw: model # No-op compile

    # Configure CallbackList mock instance
    mock_callback_list_instance = MagicMock(spec=CallbackList)
    mock_callback_list_instance.callbacks = []
    MockCallbackList.return_value = mock_callback_list_instance

    # Act
    trainer = Trainer(cfg=cfg)

    # Assert: Check if internal attributes are set correctly
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.experiment_cfg.data.batch_size == 2
    assert trainer.device == mock_initialize_device.return_value
    assert trainer.tokenizer == env["mock_tokenizer"]
    assert trainer.model is env["mock_model"]

    # Assert mocks are assigned
    assert trainer.train_dataloader is env["mock_train_loader"]
    assert trainer.val_dataloader is env["mock_val_loader"]
    assert trainer.optimizer is env["mock_optimizer"]
    assert trainer.scheduler is env["mock_scheduler"]
    assert trainer.callbacks is mock_callback_list_instance
    assert trainer.evaluator is env["mock_evaluator"]
    assert trainer.checkpoint_manager is env["mock_checkpoint_manager"]
    assert trainer.scaler is env["mock_scaler"]

    # Assert initialize functions were called correctly
    mock_initialize_device.assert_called_once()
    mock_initialize_tokenizer.assert_called_once()
    mock_initialize_model.assert_called_once()
    mock_initialize_dataloaders.assert_called_once()
    mock_initialize_optimizer.assert_called_once()
    mock_initialize_scheduler.assert_called_once()
    mock_initialize_amp_scaler.assert_called_once()
    mock_initialize_callbacks.assert_called_once()
    mock_initialize_checkpoint_manager.assert_called_once()
    mock_initialize_evaluator.assert_called_once()
    mock_compile_model.assert_called_once()
    MockCallbackList.assert_called_once_with([])

    # Assert basic properties are set correctly from config
    assert trainer.experiment_name == "test_minimal_init"
    assert trainer.experiment_cfg.data.batch_size == cfg.experiment.data.batch_size
    assert trainer.config.num_epochs == 1
    assert trainer.experiment_cfg.model._target_ == "craft.models.transformer.TransformerModel"
    assert trainer.experiment_cfg.optimizer.lr == 1e-4

    # Assert that the main training loop object was created
    assert isinstance(trainer.training_loop, TrainingLoop) 