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
from craft.training.training_loop import TrainingLoop # Add this import

# Dummy class needed for callback test target within test_init_with_optionals
# (even though the test file containing the definition might change later)
class MockCallbackTarget:
    pass

# =================================
# Initialization Tests - Basic & Optionals
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
def test_init_with_optionals_no_callbacks(
    # Mocks passed in reverse order
    MockCallbackList,
    mock_compile_model,
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
    setup_optional_trainer_env,
):
    """Test Trainer initialization with optional components using initialize patches."""
    # Arrange
    env = setup_optional_trainer_env
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
    try:
        trainer = Trainer(cfg=cfg)
    except Exception as e:
        pytest.fail(f"Trainer initialization with optionals failed unexpectedly: {e}")

    # Assert: Check internal attributes
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.experiment_cfg.data.batch_size == 4
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
    assert trainer.experiment_name == "test_optional_init"
    assert trainer.experiment_cfg.data.batch_size == cfg.experiment.data.batch_size
    assert trainer.config.num_epochs == 2
    assert trainer.experiment_cfg.model._target_ == "craft.models.transformer.TransformerModel"
    assert trainer.experiment_cfg.optimizer.lr == 1e-4

    # Assert that the main training loop object was created
    assert isinstance(trainer.training_loop, TrainingLoop)


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
def test_init_with_optionals_no_callbacks_no_model(
    # Mocks passed in reverse order
    MockCallbackList,
    mock_compile_model,
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
    setup_optional_trainer_env,
):
    """Test Trainer initialization with optional components using initialize patches."""
    # Arrange
    env = setup_optional_trainer_env
    cfg = env["experiment_config"]

    # Configure the mocks to return values from the env fixture
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_tokenizer.return_value = env["mock_tokenizer"]
    mock_initialize_model.return_value = None
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
    try:
        trainer = Trainer(cfg=cfg)
    except Exception as e:
        pytest.fail(f"Trainer initialization with optionals failed unexpectedly: {e}")

    # Assert: Check internal attributes
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.experiment_cfg.data.batch_size == 4
    assert trainer.device == mock_initialize_device.return_value
    assert trainer.tokenizer == env["mock_tokenizer"]
    assert trainer.model is None

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
    assert trainer.experiment_name == "test_optional_init"
    assert trainer.experiment_cfg.data.batch_size == cfg.experiment.data.batch_size
    assert trainer.config.num_epochs == 2
    assert trainer.experiment_cfg.model._target_ == "craft.models.transformer.TransformerModel"
    assert trainer.experiment_cfg.optimizer.lr == 1e-4

    # Assert that the main training loop object was created
    assert isinstance(trainer.training_loop, TrainingLoop)


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
def test_init_with_optionals_no_callbacks_no_model_no_optimizer(
    # Mocks passed in reverse order
    MockCallbackList,
    mock_compile_model,
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
    setup_optional_trainer_env,
):
    """Test Trainer initialization with optional components using initialize patches."""
    # Arrange
    env = setup_optional_trainer_env
    cfg = env["experiment_config"]

    # Configure the mocks to return values from the env fixture
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_tokenizer.return_value = env["mock_tokenizer"]
    mock_initialize_model.return_value = None
    mock_initialize_dataloaders.return_value = (env["mock_train_loader"], env["mock_val_loader"]) # Tuple
    mock_initialize_optimizer.return_value = None
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
    try:
        trainer = Trainer(cfg=cfg)
    except Exception as e:
        pytest.fail(f"Trainer initialization with optionals failed unexpectedly: {e}")

    # Assert: Check internal attributes
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.experiment_cfg.data.batch_size == 4
    assert trainer.device == mock_initialize_device.return_value
    assert trainer.tokenizer == env["mock_tokenizer"]
    assert trainer.model is None

    # Assert mocks are assigned
    assert trainer.train_dataloader is env["mock_train_loader"]
    assert trainer.val_dataloader is env["mock_val_loader"]
    assert trainer.optimizer is None
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
    assert trainer.experiment_name == "test_optional_init"
    assert trainer.experiment_cfg.data.batch_size == cfg.experiment.data.batch_size
    assert trainer.config.num_epochs == 2
    assert trainer.experiment_cfg.model._target_ == "craft.models.transformer.TransformerModel"
    assert trainer.experiment_cfg.optimizer.lr == 1e-4

    # Assert that the main training loop object was created
    assert isinstance(trainer.training_loop, TrainingLoop)


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
def test_init_with_optionals_no_callbacks_no_model_no_optimizer_no_scheduler(
    # Mocks passed in reverse order
    MockCallbackList,
    mock_compile_model,
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
    setup_optional_trainer_env,
):
    """Test Trainer initialization with optional components using initialize patches."""
    # Arrange
    env = setup_optional_trainer_env
    cfg = env["experiment_config"]

    # Configure the mocks to return values from the env fixture
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_tokenizer.return_value = env["mock_tokenizer"]
    mock_initialize_model.return_value = None
    mock_initialize_dataloaders.return_value = (env["mock_train_loader"], env["mock_val_loader"]) # Tuple
    mock_initialize_optimizer.return_value = None
    mock_initialize_scheduler.return_value = None
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
    try:
        trainer = Trainer(cfg=cfg)
    except Exception as e:
        pytest.fail(f"Trainer initialization with optionals failed unexpectedly: {e}")

    # Assert: Check internal attributes
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.experiment_cfg.data.batch_size == 4
    assert trainer.device == mock_initialize_device.return_value
    assert trainer.tokenizer == env["mock_tokenizer"]
    assert trainer.model is None

    # Assert mocks are assigned
    assert trainer.train_dataloader is env["mock_train_loader"]
    assert trainer.val_dataloader is env["mock_val_loader"]
    assert trainer.optimizer is None
    assert trainer.scheduler is None
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
    assert trainer.experiment_name == "test_optional_init"
    assert trainer.experiment_cfg.data.batch_size == cfg.experiment.data.batch_size
    assert trainer.config.num_epochs == 2
    assert trainer.experiment_cfg.model._target_ == "craft.models.transformer.TransformerModel"
    assert trainer.experiment_cfg.optimizer.lr == 1e-4

    # Assert that the main training loop object was created
    assert isinstance(trainer.training_loop, TrainingLoop)


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
def test_init_with_optionals_no_callbacks_no_model_no_optimizer_no_scheduler_no_scaler(
    # Mocks passed in reverse order
    MockCallbackList,
    mock_compile_model,
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
    setup_optional_trainer_env,
):
    """Test Trainer initialization with optional components using initialize patches."""
    # Arrange
    env = setup_optional_trainer_env
    cfg = env["experiment_config"]

    # Configure the mocks to return values from the env fixture
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_tokenizer.return_value = env["mock_tokenizer"]
    mock_initialize_model.return_value = None
    mock_initialize_dataloaders.return_value = (env["mock_train_loader"], env["mock_val_loader"]) # Tuple
    mock_initialize_optimizer.return_value = None
    mock_initialize_scheduler.return_value = None
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
    try:
        trainer = Trainer(cfg=cfg)
    except Exception as e:
        pytest.fail(f"Trainer initialization with optionals failed unexpectedly: {e}")

    # Assert: Check internal attributes
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.experiment_cfg.data.batch_size == 4
    assert trainer.device == mock_initialize_device.return_value
    assert trainer.tokenizer == env["mock_tokenizer"]
    assert trainer.model is None

    # Assert mocks are assigned
    assert trainer.train_dataloader is env["mock_train_loader"]
    assert trainer.val_dataloader is env["mock_val_loader"]
    assert trainer.optimizer is None
    assert trainer.scheduler is None
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
    assert trainer.experiment_name == "test_optional_init"
    assert trainer.experiment_cfg.data.batch_size == cfg.experiment.data.batch_size
    assert trainer.config.num_epochs == 2
    assert trainer.experiment_cfg.model._target_ == "craft.models.transformer.TransformerModel"
    assert trainer.experiment_cfg.optimizer.lr == 1e-4

    # Assert that the main training loop object was created
    assert isinstance(trainer.training_loop, TrainingLoop)