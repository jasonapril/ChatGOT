"""
Integration tests for the checkpoint save/load/resume cycle.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.amp.grad_scaler import GradScaler
import os
from pathlib import Path
from unittest.mock import MagicMock # Keep for mocking dataloader/tokenizer if needed

# Components under test / dependencies
from craft.training.checkpointing import CheckpointManager, TrainingState
from craft.config.schemas import TrainingConfig # Assuming default config is sufficient
from craft.training.callbacks import CallbackList

# Helper function to compare state dicts (can be moved to a test util module)
def assert_state_dicts_equal(state_dict1, state_dict2):
    """Asserts two state dictionaries are equal, handling tensors with allclose."""
    assert state_dict1.keys() == state_dict2.keys(), "State dict keys do not match."
    for key in state_dict1:
        val1 = state_dict1[key]
        val2 = state_dict2[key]
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            # Use torch.allclose for tensor comparison with default tolerances
            assert torch.allclose(val1, val2), f"Tensor mismatch in key: {key}"
        elif isinstance(val1, dict) and isinstance(val2, dict):
            # Recursively check nested dictionaries (like optimizer state)
            assert_state_dicts_equal(val1, val2)
        else:
            # For non-tensor, non-dict types, use standard equality
            assert val1 == val2, f"Value mismatch in key: {key} ({type(val1)} vs {type(val2)})"

# --- Fixtures --- #

@pytest.fixture
def simple_model():
    """Provides a very simple linear model."""
    return nn.Linear(10, 2) # Simple enough to check weights

@pytest.fixture
def simple_optimizer(simple_model):
    """Provides a basic AdamW optimizer."""
    return optim.AdamW(simple_model.parameters(), lr=1e-3)

@pytest.fixture
def simple_scheduler(simple_optimizer):
    """Provides a basic StepLR scheduler."""
    return StepLR(simple_optimizer, step_size=1, gamma=0.9)

@pytest.fixture
def simple_scaler():
    """Provides a basic GradScaler."""
    return GradScaler(enabled=True, device="cpu")

@pytest.fixture
def checkpoint_components(simple_model, simple_optimizer, simple_scheduler, simple_scaler):
    """Bundles the core components needed for checkpointing tests."""
    # Create a minimal valid TrainingConfig for the test
    minimal_training_config = TrainingConfig(
        batch_size=4, # Provide a value for the required field
        max_steps=10, # Provide a value (adjust if needed by other components)
        # Add other required fields from TrainingConfig schema if validation fails again
        # Optional fields with defaults can be omitted unless specific values are needed
        log_interval=1,
        save_interval=0, # Disable epoch-based saving for this test
        save_steps_interval=0, # Disable step-based saving for this test
        time_save_interval_seconds=0, # Disable time-based saving for this test
        eval_interval=0, # Disable eval for this test
    )
    return {
        "model": simple_model,
        "optimizer": simple_optimizer,
        "scheduler": simple_scheduler,
        "scaler": simple_scaler,
        "config": minimal_training_config.model_dump(), # Use the valid config
        "callbacks": CallbackList([]), # Empty callbacks list
        "device": torch.device("cpu"), # Use CPU for simplicity
        "tokenizer": None, # No tokenizer needed for this integrity test
    }

# --- Test Class --- #

class TestCheckpointIntegration:

    def test_checkpoint_save_load_resume_integrity(self, checkpoint_components, tmp_path):
        """Tests saving and loading the full training state for resume."""
        # 1. Initial State & Checkpoint Manager
        initial_components = checkpoint_components
        exp_name = "test_integrity_exp"
        initial_cm = CheckpointManager(
            str(tmp_path), # Pass checkpoint_dir positionally
            model=initial_components["model"],
            optimizer=initial_components["optimizer"],
            scheduler=initial_components["scheduler"],
            scaler=initial_components["scaler"],
            callbacks=initial_components["callbacks"],
            device=initial_components["device"],
            tokenizer=initial_components["tokenizer"],
            config=initial_components["config"],
            experiment_name=exp_name,
        )
        # # Manually set the dir for this test - NO LONGER NEEDED
        # initial_cm.checkpoint_dir = tmp_path
        
        # 2. Simulate Training Steps (Manually for simplicity)
        initial_model = initial_components["model"]
        initial_optimizer = initial_components["optimizer"]
        initial_scheduler = initial_components["scheduler"]
        initial_scaler = initial_components["scaler"]
        
        initial_optimizer.zero_grad()
        # Dummy forward/backward
        dummy_input = torch.randn(4, 10)
        dummy_output = initial_model(dummy_input)
        dummy_loss = dummy_output.mean()
        initial_scaler.scale(dummy_loss).backward()
        # Step components
        initial_scaler.step(initial_optimizer)
        initial_scaler.update()
        initial_scheduler.step()
        
        simulated_epoch = 1
        simulated_global_step = 1
        simulated_best_val = 1.23 # Dummy value

        # 3. Get State Pre-Save
        model_state_pre = initial_model.state_dict()
        optimizer_state_pre = initial_optimizer.state_dict()
        scheduler_state_pre = initial_scheduler.state_dict()
        scaler_state_pre = initial_scaler.state_dict()
        
        # 4. Save Checkpoint
        save_filename = "test_checkpoint_integrity.pt"
        training_state_to_save = TrainingState(
            epoch=simulated_epoch,
            global_step=simulated_global_step,
            model_state_dict=model_state_pre, # Save the captured state
            optimizer_state_dict=optimizer_state_pre,
            scheduler_state_dict=scheduler_state_pre,
            scaler_state_dict=scaler_state_pre,
            config=initial_components["config"],
            best_val_metric=simulated_best_val,
            # Other fields can be None/default for this test
        )
        initial_cm.save_checkpoint(state=training_state_to_save, filename=save_filename)
        checkpoint_path = tmp_path / save_filename
        assert checkpoint_path.exists()

        # 5. Reset/Re-create Components
        # Create NEW instances to simulate loading into a fresh run
        new_model = nn.Linear(10, 2)
        new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-3)
        new_scheduler = StepLR(new_optimizer, step_size=1, gamma=0.9)
        new_scaler = GradScaler(enabled=True, device="cpu")
        new_callbacks = CallbackList([])

        # 6. Create New Checkpoint Manager
        new_cm = CheckpointManager(
            str(tmp_path), # Pass checkpoint_dir positionally
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=new_scaler,
            callbacks=new_callbacks,
            device=initial_components["device"],
            tokenizer=None,
            config={},
            experiment_name=exp_name,
        )
        # # Manually set the dir for this test - NO LONGER NEEDED
        # new_cm.checkpoint_dir = tmp_path

        # 7. Load Checkpoint
        loaded_training_state = new_cm.load_checkpoint(path_specifier=str(checkpoint_path))

        # 8. Assertions
        assert loaded_training_state is not None
        assert isinstance(loaded_training_state, TrainingState)

        # --- Apply Loaded State --- #
        new_model.load_state_dict(loaded_training_state.model_state_dict)
        if loaded_training_state.optimizer_state_dict:
            new_optimizer.load_state_dict(loaded_training_state.optimizer_state_dict)
        if loaded_training_state.scheduler_state_dict and new_scheduler:
            new_scheduler.load_state_dict(loaded_training_state.scheduler_state_dict)
        if loaded_training_state.scaler_state_dict and new_scaler:
            new_scaler.load_state_dict(loaded_training_state.scaler_state_dict)
        # TODO: Add callback state loading if needed: new_callbacks.load_state_dict(...)

        # Check trainer state variables
        assert loaded_training_state.epoch == simulated_epoch
        assert loaded_training_state.global_step == simulated_global_step
        assert loaded_training_state.best_val_metric == simulated_best_val

        # Check component states
        # Model state was loaded into new_model by new_cm.load_checkpoint
        assert_state_dicts_equal(new_model.state_dict(), model_state_pre)
        # Optimizer state was loaded into new_optimizer
        assert_state_dicts_equal(new_optimizer.state_dict(), optimizer_state_pre)
        # Scheduler state was loaded into new_scheduler
        assert_state_dicts_equal(new_scheduler.state_dict(), scheduler_state_pre)
        # Scaler state was loaded into new_scaler
        assert_state_dicts_equal(new_scaler.state_dict(), scaler_state_pre) 