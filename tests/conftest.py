"""
Configuration and fixtures for pytest.

Shared fixtures defined here are available to all tests.
"""

import pytest
from unittest.mock import MagicMock
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize_config_dir
from pathlib import Path

from craft.training.generation import TextGenerator

# Define the path to the test configs relative to this file
# Assuming conftest.py is in the tests/ directory
CONFIG_DIR = Path(__file__).parent / "fixtures" / "configs"


@pytest.fixture(scope="function") # Use 'function' scope for isolation unless setup is very slow
def minimal_cfg() -> DictConfig:
    """Loads the minimal experiment configuration using Hydra."""
    abs_config_dir = CONFIG_DIR.resolve()
    # Use initialize_config_dir, safe for running within pytest
    # It prevents Hydra from changing the CWD globally for all tests
    with initialize_config_dir(config_dir=str(abs_config_dir), job_name="test_minimal_cfg"):
        # Important: Use strict=False if Pydantic schemas aren't fully defined yet or configs have extra keys
        cfg = hydra.compose(config_name="minimal_experiment", strict=False)
        # Optional: Resolve interpolations if tests need final values immediately
        # OmegaConf.resolve(cfg)
        return cfg

# --- Placeholder for other fixtures using minimal_cfg --- #
# @pytest.fixture
# def minimal_trainer_components(minimal_cfg: DictConfig):
#     # Instantiate model, data parts etc. from minimal_cfg using hydra.utils.instantiate
#     # Requires dummy data/models referenced in YAMLs to exist
#     components = {}
#     components['model'] = hydra.utils.instantiate(minimal_cfg.model)
#     # ... instantiate others ...
#     # Requires setup for dummy data referenced in minimal_data.yaml
#     # components['data'] = hydra.utils.instantiate(minimal_cfg.data)
#     return components

# @pytest.fixture
# def minimal_trainer_instance(minimal_trainer_components):
#     # Instantiate the actual Trainer using the minimal components
#     # from craft.training import Trainer
#     # return Trainer(...)
#     pass

# --- Existing Mock Fixtures --- #

@pytest.fixture
def mock_optimizer():
    """Provides a mock optimizer with necessary attributes."""
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    # Ensure param_groups is a list containing a dict with 'lr'
    optimizer.param_groups = [{'lr': 0.01}]
    # Mock step method if needed
    optimizer.step = MagicMock()
    # Mock zero_grad method if needed
    optimizer.zero_grad = MagicMock()
    return optimizer

@pytest.fixture
def mock_trainer(mock_optimizer):
    """Provides a mock Trainer instance with common attributes needed by callbacks."""
    trainer = MagicMock()
    trainer.model = MagicMock(spec=torch.nn.Module)
    trainer.model.training = True # Assume model starts in training mode
    trainer.optimizer = mock_optimizer
    trainer.device = torch.device("cpu")
    trainer.config = MagicMock() # Mock config object
    trainer.logger = MagicMock(spec=logging.Logger)
    # Crucially, add the attribute expected by EarlyStopping
    trainer._stop_training = False
    # Add other attributes commonly used/checked
    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.dataset = MagicMock() # Mock dataset
    trainer.val_dataloader = None
    trainer.scheduler = None
    trainer.tokenizer = MagicMock() # Add a mock tokenizer
    trainer.callbacks = MagicMock() # Mock CallbackList
    trainer.checkpoint_manager = MagicMock()
    trainer.epoch = 0
    trainer.global_step = 0

    # Reset relevant state before yielding to the test
    trainer._stop_training = False

    yield trainer # Yield the trainer to the test

    # Any teardown if needed after the test runs

@pytest.fixture
def mock_text_generator():
    """Provides a mock TextGenerator instance."""
    generator = MagicMock(spec=TextGenerator) # Assuming TextGenerator is imported or accessible
    # Mock the method used by the callback
    generator.generate_text = MagicMock(return_value=["Mock generated text."])
    return generator

# Add other shared fixtures below if needed 