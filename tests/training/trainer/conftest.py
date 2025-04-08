"""Configuration for pytest in the Trainer initialization tests."""

import pytest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler as CudaGradScaler # Alias to avoid conflict
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError

# Assuming these types are importable from their respective modules
from craft.config.schemas import TrainingConfig, LanguageModelConfig
from craft.data.tokenizers.base import Tokenizer
from craft.models.base import LanguageModel # Assuming LanguageModel is the base or relevant type
from craft.training.callbacks.base import Callback, CallbackList
from craft.training.evaluation import Evaluator
from craft.training.checkpointing import CheckpointManager
# from craft.training.trainer import Trainer # Not needed directly in conftest

# Dummy classes for patching targets if they don't exist or for isolation
class MockTokenizerTarget:
    pass
class MockTrainDatasetTarget:
    pass
class MockValDatasetTarget:
    pass
class MockModelTarget:
    pass
class MockOptimizerTarget:
    pass
class MockSchedulerTarget:
    pass
class MockCallbackTarget:
    pass
class MockEvaluatorTarget:
    pass
class MockCheckpointManagerTarget:
    pass

# =================================
# Fixtures
# =================================

@pytest.fixture
def setup_minimal_trainer_env():
    """Provides a minimal DictConfig and mock instances for basic Trainer init."""
    # Basic training config
    training_args = {
        "_target_": "craft.config.schemas.TrainingConfig", "batch_size": 2, "num_epochs": 1,
        "use_amp": False, "gradient_accumulation_steps": 1, "log_interval": 10,
        "eval_interval": 50, "save_interval": 100, "keep_last": 1,
        "resume_from_checkpoint": None, "log_level": "INFO", "seed": 42,
        "device": "cpu", "max_steps": None, "learning_rate": 1e-4,
        "max_grad_norm": 1.0, "torch_compile": False, "sample_max_new_tokens": 10,
        "sample_temperature": 0.7, "sample_start_text": "Hello", "val_metric": "loss",
        "time_save_interval_seconds": 0, "time_eval_interval_seconds": 0,
        "mixed_precision": False, "save_steps_interval": 0, "compile_model": False,
    }
    try:
        expected_pydantic_config = TrainingConfig(**training_args)
    except ValidationError as e:
        pytest.fail(f"Failed to create valid TrainingConfig in minimal fixture: {e}")

    # Basic Experiment Config Structure (using placeholders)
    exp_dict = {
        'experiment': {
            'experiment_name': 'test_minimal_init',
            'device': training_args['device'],
            'training': OmegaConf.create(training_args),
            'model': OmegaConf.create({
                '_target_': 'craft.models.transformer.TransformerModel',
                'config': { 'vocab_size': 50 } # Example minimal model config
            }),
            'data': OmegaConf.create({
                'tokenizer': {'_target_': 'craft.data.tokenizers.char.CharTokenizer'},
                'batch_size': training_args['batch_size'],
                'num_workers': 0,
                'datasets': {
                    'train': {'_target_': 'placeholder.train.DatasetTarget'}
                    # No val dataset in minimal config
                }
            }),
            'optimizer': OmegaConf.create({
                '_target_': 'torch.optim.AdamW',
                'lr': training_args['learning_rate']
            }),
            # No scheduler, callbacks, evaluation, checkpointing in minimal config
            'scheduler': None,
            'callbacks': None,
            'evaluation': None,
            'checkpointing': None,
        }
    }
    experiment_config_node = OmegaConf.create(exp_dict)

    # Mocks for required components
    mock_tokenizer = MagicMock(spec=Tokenizer)
    mock_tokenizer.get_vocab_size.return_value = 50
    mock_model = MagicMock(spec=LanguageModel)
    mock_model.config = MagicMock(vocab_size=50)
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))] # Need parameters for optimizer
    mock_model.to.return_value = mock_model # Make .to() chainable
    mock_train_dataset = MagicMock() # No spec needed for simple length check
    mock_train_dataset.__len__ = MagicMock(return_value=20)
    mock_train_loader = DataLoader(mock_train_dataset, batch_size=training_args['batch_size']) # Use real loader for type check
    mock_optimizer = MagicMock(spec=AdamW)
    mock_optimizer.param_groups = [{'lr': training_args['learning_rate']}] # Mimic optimizer state

    return {
        "experiment_config": experiment_config_node,
        "expected_training_config": expected_pydantic_config,
        "mock_tokenizer": mock_tokenizer,
        "mock_model": mock_model,
        "mock_train_dataset": mock_train_dataset, # Dataset returned by instantiate
        "mock_train_loader": mock_train_loader, # Final loader attribute
        "mock_optimizer": mock_optimizer,
        # No optional mocks in minimal setup
        "mock_val_dataset": None,
        "mock_val_loader": None,
        "mock_scheduler": None,
        "mock_callback": None,
        "mock_evaluator": None,
        "mock_checkpoint_manager": None,
        "mock_scaler": None
    }


@pytest.fixture
def setup_optional_trainer_env():
    """Provides a DictConfig with optional components and mock instances."""
    # Use different settings for optionals, e.g., use_amp=True
    training_args = {
        "_target_": "craft.config.schemas.TrainingConfig", "batch_size": 4, "num_epochs": 2,
        "use_amp": True, "gradient_accumulation_steps": 1, "log_interval": 10,
        "eval_interval": 50, "save_interval": 100, "keep_last": 2,
        "resume_from_checkpoint": None, "log_level": "DEBUG", "seed": 42,
        "device": "cpu", "max_steps": None, "learning_rate": 1e-4,
        "max_grad_norm": 1.0, "torch_compile": False, "sample_max_new_tokens": 10,
        "sample_temperature": 0.7, "sample_start_text": "Hello", "val_metric": "loss",
        "time_save_interval_seconds": 0, "time_eval_interval_seconds": 0,
        "mixed_precision": False, "save_steps_interval": 0, "compile_model": False,
    }
    try:
        expected_pydantic_config = TrainingConfig(**training_args)
    except ValidationError as e:
        pytest.fail(f"Failed to create valid TrainingConfig in optional fixture: {e}")

    # Experiment Config Structure with optional targets (use real targets where possible)
    exp_dict = {
        'experiment': {
            'experiment_name': 'test_optional_init',
            'device': training_args['device'],
            'training': OmegaConf.create(training_args),
            'model': OmegaConf.create({
                '_target_': 'craft.models.transformer.TransformerModel', # Real target
                'config': { 'vocab_size': 50 }
            }),
            'data': OmegaConf.create({
                'tokenizer': {'_target_': 'craft.data.tokenizers.char.CharTokenizer'}, # Example real target
                'batch_size': training_args['batch_size'],
                'num_workers': 0,
                'datasets': {
                    'train': {'_target_': 'placeholder.train.DatasetTarget'}, # Placeholder target
                    'val': {'_target_': 'placeholder.val.DatasetTarget'} # Placeholder target
                }
            }),
            'optimizer': OmegaConf.create({
                '_target_': 'torch.optim.AdamW', # Real target
                'lr': training_args['learning_rate']
            }),
            'scheduler': { # Use real target
                '_target_': 'torch.optim.lr_scheduler.StepLR',
                'step_size': 1
            },
            'callbacks': { # Keep mock target for simplicity here
                'mock_cb': { '_target_': 'tests.training.trainer.test_trainer_init.MockCallbackTarget' } # Target in test file
            },
            'evaluation': { # Example real target
                '_target_': 'craft.training.evaluation.Evaluator'
            },
            'checkpointing': { # Example real target
                '_target_': 'craft.training.checkpointing.CheckpointManager'
            },
        }
    }
    experiment_config_node = OmegaConf.create(exp_dict)

    # Mocks for required and optional components
    mock_tokenizer = MagicMock(spec=Tokenizer)
    mock_tokenizer.get_vocab_size.return_value = 50
    mock_model = MagicMock(spec=LanguageModel)
    mock_model.config = MagicMock(vocab_size=50)
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    mock_model.to.return_value = mock_model
    mock_train_dataset = MagicMock() # No spec
    mock_train_dataset.__len__ = MagicMock(return_value=20)
    mock_val_dataset = MagicMock() # No spec
    mock_val_dataset.__len__ = MagicMock(return_value=10)
    mock_train_loader = DataLoader(mock_train_dataset, batch_size=training_args['batch_size'])
    mock_val_loader = DataLoader(mock_val_dataset, batch_size=training_args['batch_size'])
    mock_optimizer = MagicMock(spec=AdamW)
    mock_optimizer.param_groups = [{'lr': training_args['learning_rate']}]
    mock_scheduler = MagicMock(spec=_LRScheduler)
    mock_callback = MagicMock(spec=Callback)
    mock_evaluator = MagicMock(spec=Evaluator)
    mock_checkpoint_manager = MagicMock(spec=CheckpointManager)
    mock_scaler = MagicMock(spec=CudaGradScaler)

    return {
        "experiment_config": experiment_config_node,
        "expected_training_config": expected_pydantic_config,
        "mock_tokenizer": mock_tokenizer,
        "mock_model": mock_model,
        "mock_train_dataset": mock_train_dataset, # Dataset returned by instantiate
        "mock_val_dataset": mock_val_dataset,   # Dataset returned by instantiate
        "mock_train_loader": mock_train_loader, # Final loader attribute
        "mock_val_loader": mock_val_loader,     # Final loader attribute
        "mock_optimizer": mock_optimizer,
        "mock_scheduler": mock_scheduler,
        "mock_callback": mock_callback,   # Instance returned by MockCallbackTarget
        "mock_evaluator": mock_evaluator,
        "mock_checkpoint_manager": mock_checkpoint_manager,
        "mock_scaler": mock_scaler        # Needed because use_amp=True
    }


# =================================
# Side Effect Function for Instantiate
# =================================

def create_instantiate_side_effect(env):
    """Creates a side_effect function for patching craft.training.trainer.instantiate."""
    def instantiate_side_effect(*args, **kwargs): # More general signature
        # Try to find the config node or target string
        config = None
        target = None
        if args:
            if isinstance(args[0], (DictConfig, dict)):
                config = args[0]
                target = config.get('_target_')
            elif isinstance(args[0], str):
                target = args[0]
        elif '_target_' in kwargs:
             target = kwargs.get('_target_')
             config = kwargs # Treat kwargs as the config dict

        # print(f"DEBUG side_effect: target={target}, config_type={type(config)}, args={args}, kwargs={kwargs}")

        # Return mocks based on target
        if target == 'craft.data.tokenizers.char.CharTokenizer':
            return env["mock_tokenizer"]
        elif target == 'placeholder.train.DatasetTarget':
            return env["mock_train_dataset"]
        elif target == 'placeholder.val.DatasetTarget':
            return env.get("mock_val_dataset")
        # Model is NOT instantiated via this patch
        elif target == 'torch.optim.AdamW':
            return env["mock_optimizer"]
        elif target == 'torch.optim.lr_scheduler.StepLR':
            return env.get("mock_scheduler")
        elif target == 'tests.training.trainer.test_trainer_init.MockCallbackTarget':
            return env.get("mock_callback")
        elif target == 'craft.training.evaluation.Evaluator':
            return env.get("mock_evaluator")
        elif target == 'craft.training.checkpointing.CheckpointManager':
            return env.get("mock_checkpoint_manager")
        else:
            if target:
                raise ValueError(f"Test mock setup: Unhandled instantiate target: {target} with config {config}, args {args}, kwargs {kwargs}")
            else:
                 raise ValueError(f"Test mock setup: instantiate called without discernible target: args={args}, kwargs={kwargs}")
    return instantiate_side_effect 