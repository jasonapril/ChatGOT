"""
Configuration and fixtures for pytest.

Shared fixtures defined here are available to all tests.
"""

import pytest
from unittest.mock import MagicMock
import torch
import logging

# Example fixture (can be removed if not needed initially)
# @pytest.fixture(scope="session")
# def shared_resource():
#     print("\nSetting up shared resource")
#     yield "resource_data"
#     print("\nTearing down shared resource")

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