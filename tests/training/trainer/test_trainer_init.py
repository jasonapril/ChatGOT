import pytest
import torch
from unittest.mock import MagicMock, patch, ANY, call
import logging
from pydantic import ValidationError
from omegaconf import OmegaConf, DictConfig

# Import the class to test
from craft.training.trainer import Trainer
from craft.training.callbacks import CallbackList
from craft.training.checkpointing import CheckpointManager
from torch.cuda.amp import GradScaler as CudaGradScaler # Alias to avoid conflict if torch.amp is used
from craft.config.schemas import TrainingConfig, AppConfig, LanguageModelConfig, CheckpointConfig, DataConfig, OptimizerConfig, SchedulerConfig, AnyModelConfig, ExperimentConfig, LogConfig, CallbacksConfig, AdamWOptimizerConfig # Import necessary configs
from craft.data.tokenizers.base import Tokenizer # Import base Tokenizer

# Define a simple model config dict fixture for testing
@pytest.fixture
def simple_model_config_dict():
    # Example config, adjust fields as needed for tests
    cfg = LanguageModelConfig(architecture='transformer', vocab_size=50, d_model=32, n_head=2, n_layers=1)
    # Return as dict, mimicking how it might be passed from train.py
    return cfg.model_dump()

# Mock Experiment Config Fixture
@pytest.fixture
def mock_experiment_config() -> DictConfig:
    """Creates a mock OmegaConf DictConfig representing experiment config."""
    conf_dict = {
        'data': {
            '_target_': 'mock_data_factory', # Target for dataloaders
            'tokenizer': {
                '_target_': 'mock_tokenizer_factory' # Target for tokenizer
            },
            'batch_size': 4,
            'num_workers': 0,
            'block_size': 32,
            'datasets': { # Include datasets structure for fallback testing if needed
                'train': {
                    'dataset': {'_target_': 'mock_train_dataset'},
                    'dataloader': {}
                },
                'val': {
                     'dataset': {'_target_': 'mock_val_dataset'},
                     'dataloader': {}
                 }
             }
        },
        'model': {
            # Model config comes separately as model_config_dict,
            # but keep a placeholder here if needed for other tests
            'architecture': 'transformer',
            '_target_': 'craft.models.transformer.TransformerModel' # Added for completeness
        },
        'optimizer': {
            '_target_': 'mock_optimizer_factory',
            'lr': 1e-4,
            'weight_decay': 0.01
        },
        'scheduler': None, # Minimal case: no scheduler
        'callbacks': None # Minimal case: no callbacks
    }
    return OmegaConf.create(conf_dict)

# Full Mock Experiment Config Fixture (with scheduler and callbacks)
@pytest.fixture
def mock_full_experiment_config() -> DictConfig:
    """Creates a mock OmegaConf DictConfig with scheduler and callback configs."""
    conf_dict = {
        'data': {
            '_target_': 'mock_data_factory',
            'tokenizer': { '_target_': 'mock_tokenizer_factory' },
            'batch_size': 8, 'num_workers': 2, 'block_size': 64,
            'datasets': { 'train': {'dataset': {'_target_': 'mock_train_dataset'}},
                          'val': {'dataset': {'_target_': 'mock_val_dataset'}}}
        },
        'model': { 'architecture': 'transformer', '_target_': 'craft.models.transformer.TransformerModel' },
        'optimizer': {
            '_target_': 'mock_optimizer_factory',
            'lr': 5e-5
        },
        'scheduler': {
            '_target_': 'mock_scheduler_factory',
            'T_max': 10000 # Example required param
        },
        'callbacks': {
            'mock_cb_1': { '_target_': 'mock_callback_factory_1' },
            'mock_cb_2': { '_target_': 'mock_callback_factory_2' }
        }
    }
    return OmegaConf.create(conf_dict)

# --- Test Trainer Initialization --- #
@pytest.fixture
def minimal_training_config():
    """Provides a TrainingConfig with only essential fields."""
    # Minimal config structure based on Trainer's needs
    return TrainingConfig(
        batch_size=4,
        num_epochs=1,
        learning_rate=1e-4,
        # Required by CheckpointManager init indirectly via Trainer
        log_interval=10,
        checkpoint_dir="./checkpoints_init_minimal",
        # Other fields have defaults
    )

@pytest.fixture
def minimal_model_config_dict():
    """Provides a minimal model config dictionary."""
    # Match LanguageModelConfig structure expected by Trainer (as dict)
    return {
        "_target_": "craft.models.simple_rnn.SimpleRNN", # Example target
        "vocab_size": 50, # Must be provided
        "embedding_dim": 16,
        "hidden_dim": 32,
        "num_layers": 1
    }

@pytest.fixture
def minimal_experiment_config_node(minimal_training_config, minimal_model_config_dict):
    """Provides a minimal OmegaConf node simulating experiment structure."""
    return OmegaConf.create({
        # Simulate structure Trainer expects
        "model": minimal_model_config_dict,
        # Data config needed for dataloader setup mock
        "data": {
             "_target_": "mock_data_factory", # Mock target for data
             "batch_size": minimal_training_config.batch_size
             # Add other data fields if Trainer init requires them
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": minimal_training_config.learning_rate
        },
        "scheduler": None, # Explicitly None for minimal case
        "callbacks": None,
        "checkpoints": {
            "checkpoint_dir": minimal_training_config.checkpoint_dir,
            # Add other checkpoint fields if needed by CheckpointManager init
        },
        "logging": {
            "log_interval": minimal_training_config.log_interval
        },
        "training": minimal_training_config.model_dump(), # Embed TrainingConfig dict
        "hydra": {"run": {"dir": "."}} # Mock hydra run dir
    })

@pytest.fixture
def base_trainer_setup(minimal_training_config, minimal_model_config_dict, minimal_experiment_config_node):
    """Fixture to set up the Trainer with basic mocks for init tests."""
    with patch("src.craft.training.trainer.CheckpointManager") as MockCheckpointManager, \
         patch("torch.amp.GradScaler") as MockGradScaler, \
         patch("torch.device") as mock_device, \
         patch('hydra.utils.instantiate') as mock_hydra_instantiate:

        mock_checkpoint_manager_instance = MockCheckpointManager.return_value
        mock_device.return_value = torch.device("cpu") # Default to CPU for base tests

        # Configure hydra mock side effect for base setup
        mock_train_loader = MagicMock()
        mock_optimizer = MagicMock()
        mock_hydra_instantiate.side_effect = [
            {'train': mock_train_loader, 'val': None}, # Dataloaders
            mock_optimizer, # Optimizer
            None # Scheduler
        ]


        trainer = Trainer(
            model_config=minimal_model_config_dict,
            config=minimal_training_config,
            experiment_config=minimal_experiment_config_node
        )
        yield {
            "trainer": trainer,
            "MockCheckpointManager": MockCheckpointManager,
            "mock_checkpoint_manager_instance": mock_checkpoint_manager_instance,
            "MockGradScaler": MockGradScaler,
            "mock_device": mock_device,
            "mock_hydra_instantiate": mock_hydra_instantiate # Pass the mock
        }

# --- Test Cases --- #

@patch("hydra.utils.instantiate") # Need to patch here too for direct Trainer call
def test_init_minimal_required(
    mock_hydra_instantiate_direct, # Separate mock for direct call
    minimal_training_config, minimal_model_config_dict, minimal_experiment_config_node # Use fixtures
):
    """Test Trainer initialization with only required arguments."""
    with patch("src.craft.training.trainer.CheckpointManager") as MockCheckpointManager, \
         patch("torch.amp.GradScaler") as MockGradScaler, \
         patch("torch.device") as mock_device:

        mock_device.return_value = torch.device("cpu") # Assume CPU
        mock_checkpoint_manager_instance = MockCheckpointManager.return_value

        # Configure hydra mock side effect for this specific test's Trainer call
        mock_train_loader = MagicMock()
        mock_optimizer = MagicMock()
        mock_hydra_instantiate_direct.side_effect = [
            {'train': mock_train_loader, 'val': None}, # Dataloaders
            mock_optimizer, # Optimizer
            None # Scheduler
        ]


        trainer = Trainer(
            model_config=minimal_model_config_dict,
            config=minimal_training_config,
            experiment_config=minimal_experiment_config_node
        )

        # Assertions
        assert trainer.config == minimal_training_config
        assert trainer.model_config == minimal_model_config_dict
        assert trainer.experiment_config == minimal_experiment_config_node
        assert trainer.device == torch.device("cpu") # Default or detected
        assert trainer.checkpoint_manager is mock_checkpoint_manager_instance
        MockCheckpointManager.assert_called_once() # Basic check
        # Check hydra calls
        assert mock_hydra_instantiate_direct.call_count == 3

        # Assert components not yet instantiated
        assert trainer._model is None
        assert trainer._optimizer is None
        assert trainer._scheduler is None
        assert trainer._train_dataloader is None # Should be none after init
        assert trainer._val_dataloader is None
        assert trainer._tokenizer is None
        assert trainer.scaler is not None # Scaler IS instantiated if not MPS
        assert trainer.callbacks is None # Callback list created in setup()
        assert trainer.state is not None # State object should be created

    @patch("src.craft.training.trainer.CheckpointManager")
    @patch("torch.amp.GradScaler")
    @patch("torch.device")
    @patch("hydra.utils.instantiate") # Patch hydra
    def test_init_auto_device_detection(
        mock_hydra_instantiate, # Add hydra mock arg
        mock_device, MockGradScaler, MockCheckpointManager,
        minimal_training_config, minimal_model_config_dict, minimal_experiment_config_node # Use fixtures
    ):
        """Test Trainer initializes with auto device detection."""
        mock_checkpoint_manager_instance = MockCheckpointManager.return_value
        # Mock device selection
        mock_device.return_value = torch.device("cuda") # Simulate selecting CUDA

        # Configure hydra mock side effect
        mock_train_loader = MagicMock()
        mock_optimizer = MagicMock()
        mock_hydra_instantiate.side_effect = [
            {'train': mock_train_loader, 'val': None}, # Dataloaders
            mock_optimizer, # Optimizer
            None # Scheduler
        ]


        config_auto = minimal_training_config.model_copy(update={"device": "auto"})

        # Instantiate with 'auto' device
        trainer = Trainer(
            model_config=minimal_model_config_dict,
            config=config_auto,
            experiment_config=minimal_experiment_config_node
        )

        # Assert torch.device was called with 'cuda' (or 'mps'/'cpu' depending on availability)
        # Forcing 'cuda' in this mock setup
        mock_device.assert_called_with("cuda")
        assert trainer.device == torch.device("cuda")
        MockCheckpointManager.assert_called_once()
        assert trainer.checkpoint_manager is mock_checkpoint_manager_instance
        # Assert hydra was called for dataloaders, optimizer, scheduler
        assert mock_hydra_instantiate.call_count == 3


    @patch("src.craft.training.trainer.CheckpointManager")
    @patch("torch.amp.GradScaler")
    @patch("torch.device")
    @patch("hydra.utils.instantiate") # Patch hydra
    def test_init_all_args_provided(
        mock_hydra_instantiate, # Add hydra mock arg
        mock_device, MockGradScaler, MockCheckpointManager,
        minimal_training_config, minimal_model_config_dict, minimal_experiment_config_node # Use fixtures
    ):
        """Test Trainer initialization when all optional args are provided."""
        # Setup mocks and specific device
        mock_checkpoint_manager_instance = MockCheckpointManager.return_value
        mock_specific_device = torch.device("cpu:0")
        mock_device.return_value = mock_specific_device # Simulate specific device

        # Configure hydra mock side effect
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_optimizer = MagicMock()
        mock_hydra_instantiate.side_effect = [
            {'train': mock_train_loader, 'val': mock_val_loader}, # Dataloaders
            mock_optimizer, # Optimizer
            None # Scheduler
        ]

        config_cpu = minimal_training_config.model_copy(update={"device": "cpu:0"})

        # Instantiate with specific device and experiment name
        trainer = Trainer(
            model_config=minimal_model_config_dict,
            config=config_cpu,
            experiment_config=minimal_experiment_config_node,
            # device=mock_specific_device, # Explicitly pass device (covered by config now)
            # experiment_name="test_exp_all_args"
        )

        # Assert specific device was used
        mock_device.assert_called_with("cpu:0")
        assert trainer.device == mock_specific_device
        # Assert CheckpointManager initialized correctly
        MockCheckpointManager.assert_called_once() # Check init args if needed
        assert trainer.checkpoint_manager is mock_checkpoint_manager_instance
        # Assert hydra was called for dataloaders, optimizer, scheduler
        assert mock_hydra_instantiate.call_count == 3
        # Assert GradScaler was instantiated (since device is not MPS)
        MockGradScaler.assert_called_once()
        assert isinstance(trainer.scaler, MagicMock) # Check it was assigned
    