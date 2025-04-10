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
from craft.models.base import LanguageModel # Corrected import
from craft.data.tokenizers.base import Tokenizer
from craft.training.evaluation import Evaluator
from craft.training.checkpointing import CheckpointManager

# =================================
# Full Optionals Initialization Test
# =================================
@patch("craft.training.trainer.CallbackList") # Keep CallbackList patch
@patch("torch.amp.GradScaler") # Correct patch target
def test_init_with_optionals(
    # Mocks passed in reverse order
    MockGradScaler,
    MockCallbackList,
    setup_optional_trainer_env,
):
    """Test Trainer initialization with optional components using initialize patches."""
    # Arrange
    env = setup_optional_trainer_env
    cfg = env["experiment_config"]
    mock_callback = env["mock_callback"]

    # Configure the mocks to return values from the env fixture
    # Remove mock configurations for the removed patches
    # mock_initialize_device.return_value = torch.device("cpu")
    # mock_initialize_tokenizer.return_value = env["mock_tokenizer"]
    # mock_initialize_model.return_value = env["mock_model"]
    # mock_initialize_dataloaders.return_value = (env["mock_train_loader"], env["mock_val_loader"]) # Tuple
    # mock_initialize_optimizer.return_value = env["mock_optimizer"]
    # mock_initialize_scheduler.return_value = env["mock_scheduler"]
    # mock_initialize_amp_scaler.return_value = env["mock_scaler"]
    # mock_initialize_callbacks.return_value = [mock_callback] # List with the raw callback mock
    # mock_initialize_checkpoint_manager.return_value = env["mock_checkpoint_manager"]
    # mock_initialize_evaluator.return_value = env["mock_evaluator"] # This wasn't patched
    # mock_compile_model.side_effect = lambda model, *a, **kw: model # No-op compile

    # Configure CallbackList mock instance
    mock_callback_list_instance = MagicMock(spec=CallbackList)
    mock_callback_list_instance.callbacks = [mock_callback]
    MockCallbackList.return_value = mock_callback_list_instance

    # Act
    try:
        trainer = Trainer(cfg=cfg)
    except Exception as e:
        pytest.fail(f"Trainer initialization with optionals failed unexpectedly: {e}")

    # Assert: Check internal attributes
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.experiment_cfg.data.batch_size == 4
    # Assert types instead of identity for internally created objects
    assert isinstance(trainer.device, torch.device)
    assert isinstance(trainer.tokenizer, Tokenizer) # Check type
    assert isinstance(trainer.model, LanguageModel) # Check type

    # Assert mocks/types are assigned correctly
    assert isinstance(trainer.train_dataloader, DataLoader) # Check type
    assert isinstance(trainer.val_dataloader, DataLoader) # Check type
    assert isinstance(trainer.optimizer, torch.optim.Optimizer) # Check type
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LRScheduler) # Corrected base class
    assert trainer.callbacks is mock_callback_list_instance # Keep identity check for mocked CallbackList
    assert isinstance(trainer.evaluator, Evaluator) # Check type
    assert isinstance(trainer.checkpoint_manager, CheckpointManager) # Check type
    # assert trainer.scaler is env["mock_scaler"] # Keep GradScaler mock check - This line was commented out, keep as is? Let's uncomment and use the MockGradScaler from the signature
    assert trainer.scaler is MockGradScaler.return_value # Check identity for mocked GradScaler

    # Assert initialize functions were called correctly - All removed

    # Assert basic properties are set correctly from config
    assert trainer.experiment_name == "test_optional_init"
    assert trainer.experiment_cfg.data.batch_size == cfg.experiment.data.batch_size
    assert trainer.config.num_epochs == 2
    assert trainer.experiment_cfg.model._target_ == "craft.models.transformer.TransformerModel"
    assert trainer.experiment_cfg.optimizer.lr == 1e-4

    # Assert that the main training loop object was created
    assert isinstance(trainer.training_loop, TrainingLoop) 