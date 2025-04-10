"""Configuration for pytest in the Trainer initialization tests."""

import pytest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler as CudaGradScaler # Alias to avoid conflict
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError
import hydra.utils
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Assuming these types are importable from their respective modules
from craft.config.schemas import TrainingConfig, LanguageModelConfig, AnyModelConfig, DataConfig, OptimizerConfig, SchedulerConfig # Corrected: ModelConfig -> AnyModelConfig
from craft.data.tokenizers.base import Tokenizer
from craft.models.base import LanguageModel # Assuming LanguageModel is the base or relevant type
from craft.training.callbacks.base import Callback, CallbackList
from craft.training.evaluation import Evaluator
from craft.training.checkpointing import CheckpointManager
# from tests.training.trainer.test_trainer_init_optionals import MockCallbackTarget # No longer needed
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

# Simple dummy dataset for testing instantiation
class MockDataset:
    def __init__(self, *args, **kwargs):
        # Accept any args/kwargs to avoid instantiation errors
        self._len = kwargs.get("length", 10) # Default length
        pass
    def __len__(self):
        return self._len
    def __getitem__(self, idx):
        return torch.tensor(idx), torch.tensor(idx + 1)

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

    # Basic Experiment Config Structure
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
                'block_size': 16,
                'datasets': {
                    'train': {
                        'dataset': { '_target_': 'tests.training.trainer.conftest.MockDataset', 'length': 20 }
                    },
                    'val': None,
                    'test': None
                }
            }),
            'optimizer': OmegaConf.create({
                '_target_': 'torch.optim.AdamW',
                'lr': training_args['learning_rate']
            }),
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
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    mock_model.to.return_value = mock_model
    mock_train_loader = DataLoader(list(range(20)), batch_size=training_args['batch_size'])
    mock_optimizer = MagicMock(spec=AdamW)
    mock_optimizer.param_groups = [{'lr': training_args['learning_rate']}]

    return {
        "experiment_config": experiment_config_node,
        "expected_training_config": expected_pydantic_config,
        "mock_tokenizer": mock_tokenizer,
        "mock_model": mock_model,
        "mock_train_loader": mock_train_loader,
        "mock_optimizer": mock_optimizer,
        "mock_val_dataset": None,
        "mock_val_loader": None,
        "mock_scheduler": None,
        "mock_callback": None,
        "mock_evaluator": None,
        "mock_checkpoint_manager": None,
        "mock_scaler": None
    }


@pytest.fixture
def setup_optional_trainer_env(tmp_path):
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

    # Experiment Config Structure with optional targets
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
                'tokenizer': {'_target_': 'craft.data.tokenizers.char.CharTokenizer'},
                'batch_size': training_args['batch_size'],
                'num_workers': 0,
                'block_size': 32,
                'datasets': {
                    'train': {
                         'dataset': { '_target_': 'tests.training.trainer.conftest.MockDataset', 'length': 20 }
                    },
                    'val': {
                         'dataset': { '_target_': 'tests.training.trainer.conftest.MockDataset', 'length': 10 }
                    },
                    'test': None
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
                'mock_cb': { '_target_': 'tests.helpers.mock_components.MockCallbackTarget' } # Path to new location
            },
            'evaluation': { # Example real target
                '_target_': 'craft.training.evaluation.Evaluator'
            },
            'checkpointing': { # Example real target
                '_target_': 'craft.training.checkpointing.CheckpointManager',
                'checkpoint_dir': str(tmp_path / "checkpoints")
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
    mock_train_loader = DataLoader(list(range(20)), batch_size=training_args['batch_size'])
    mock_val_loader = DataLoader(list(range(10)), batch_size=training_args['batch_size'])
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
        "mock_train_loader": mock_train_loader,
        "mock_val_loader": mock_val_loader,
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
    """Creates a side_effect function for patching hydra.utils.instantiate (or where it's used)."""
    def instantiate_side_effect(*args, **kwargs):
        # Determine config object or target string
        config_or_target = args[0] if args else None
        actual_kwargs = {**kwargs}
        if args and len(args) > 1:
             # Merge remaining positional args into kwargs if they are dicts?
             # This part is tricky. Let's assume Hydra primarily uses the first arg or kwargs.
             pass

        config_node = None
        target = None

        if isinstance(config_or_target, (DictConfig, dict)):
            config_node = config_or_target
            target = config_node.get('_target_')
        elif isinstance(config_or_target, str):
            target = config_or_target
            # If only target string is passed, kwargs contains the parameters
            config_node = kwargs # Or maybe args[1:] needs merging?

        # print(f"DEBUG instantiate_side_effect: target={target}, args[0]={type(config_or_target)}, kwargs={actual_kwargs}")

        # IMPORTANT: Check if instantiating the dataset target we just defined
        if target == 'tests.training.trainer.conftest.MockDataset':
            # Let Hydra actually instantiate this simple class
            return hydra.utils._instantiate(config_node) # Use internal instantiate to bypass patch

        # Return mocks based on other targets (keep existing logic)
        # NOTE: This logic might need adjustment if initialize_... functions change
        if target == 'craft.data.tokenizers.char.CharTokenizer':
            return env["mock_tokenizer"]
        elif target == 'torch.optim.AdamW':
            return env["mock_optimizer"]
        elif target == 'torch.optim.lr_scheduler.StepLR':
            return env.get("mock_scheduler")
        elif target == 'tests.helpers.mock_components.MockCallbackTarget':
            return env.get("mock_callback")
        elif target == 'craft.training.evaluation.Evaluator':
            return env.get("mock_evaluator")
        elif target == 'craft.training.checkpointing.CheckpointManager':
            return env.get("mock_checkpoint_manager")
        # DO NOT mock the Model target here - let initialize_model handle it
        elif target == 'craft.models.transformer.TransformerModel':
             raise ValueError("Instantiate side effect should not be called for the model target directly.")
        else:
            # Fallback for unhandled targets - allows some flexibility but can hide errors
            # Consider raising an error for stricter testing
            print(f"WARN: Unhandled instantiate target in side_effect: {target}")
            return MagicMock(name=f"mock_instantiate_{target}")

    return instantiate_side_effect 