import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch, ANY, call
import logging
import tempfile
from pathlib import Path
from pydantic import ValidationError
from craft.config.schemas import TrainingConfig, LanguageModelConfig
from craft.models.base import LanguageModel
from craft.training.callbacks import Callback, CallbackList
from craft.training.evaluation import Evaluator
from craft.training.checkpointing import CheckpointManager, TrainingState
from craft.training.progress import ProgressTracker
from craft.training.training_loop import TrainingLoop
from craft.training.trainer import Trainer
from craft.data.base import BaseDataset

# --- Mocks and Fixtures --- #

# Minimal Model Config Fixture (similar to other files)
@pytest.fixture
def minimal_model_config_dict():
    # Define a simple mock model config
    return {'_target_': 'tests.conftest.MockModel', 'architecture': 'mock'}

# --- Tests for Trainer --- #

@pytest.fixture
def setup_trainer_test_environment(minimal_model_config_dict):
    """Provides components needed for Trainer tests, including new config structure."""
    # --- Create TrainingConfig object --- #
    # Use a minimal but valid config
    training_args = {
        "batch_size": 4,
        "num_epochs": 2,
        "use_amp": False,
        "gradient_accumulation_steps": 1,
        "log_interval": 10,
        "eval_interval": 50,
        "save_interval": 100,
        "keep_last": 2,
        "checkpoint_dir": None,
        "resume_from_checkpoint": None,
        "log_level": "DEBUG",
        "seed": 42,
        "device": "cpu",
        # Add other potentially required fields with defaults if needed
        "max_steps": None,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "torch_compile": False,
        "sample_max_new_tokens": 100,
        "sample_temperature": 0.8,
        "sample_start_text": "Once upon a time",
        "val_metric": "loss", # Add default val_metric
        "time_save_interval_seconds": 0,
        "time_eval_interval_seconds": 0,
        "mixed_precision": False,
        "save_steps_interval": 0,
    }
    # --- Create validated TrainingConfig object --- #
    try:
        pydantic_config = TrainingConfig(**training_args)
    except ValidationError as e:
        pytest.fail(f"Failed to create valid TrainingConfig in fixture: {e}")

    # --- Create Experiment Config Node --- #
    experiment_conf_dict = {
        'training': OmegaConf.create(training_args),
        'model': OmegaConf.create(minimal_model_config_dict),
        'data': OmegaConf.create({ # Placeholder structure
            '_target_': 'mock_data_factory',
             'datasets': {'train': None, 'val': None, 'test': None}
        }),
        'optimizer': OmegaConf.create({'_target_': 'mock_optimizer_factory'}),
        'scheduler': None,
        'callbacks': None,
        'checkpoints': OmegaConf.create({ # Checkpoint settings nested under experiment
            'checkpoint_dir': None,
            'resume_from_checkpoint': None,
            'save_interval': 100, # Redundant? but mirroring structure
            'keep_last': 2,
            'save_steps_interval': 0
        }),
         # Add other top-level keys if Trainer expects them
        'output_dir': None,
    }
    experiment_config_node = OmegaConf.create(experiment_conf_dict)

    # --- Mock components (excluding callbacks) --- #
    # --- Mock components needed AFTER setup() for some tests --- #
    mock_model = MagicMock(spec=LanguageModel)
    mock_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))
    mock_model.config = MagicMock() # Give mock model a config attribute
    mock_model.config.vocab_size = 50 # Example
    mock_optimizer = MagicMock(spec=AdamW)
    mock_train_loader = [(torch.randn(pydantic_config.batch_size, 10), torch.randn(pydantic_config.batch_size, 10))] * 25
    mock_val_loader = [(torch.randn(pydantic_config.batch_size, 10), torch.randn(pydantic_config.batch_size, 10))] * 5
    mock_tokenizer = MagicMock() # Add mock tokenizer
    mock_tokenizer.get_vocab_size.return_value = 50 # Match model config
    mock_scaler = MagicMock(spec=torch.cuda.amp.GradScaler)
    
    return {
        "training_config": pydantic_config,
        "model_config_dict": minimal_model_config_dict,
        "experiment_config": experiment_config_node,
        # Mocks for post-setup state simulation
        "mock_model": mock_model,
        "mock_optimizer": mock_optimizer,
        "mock_train_loader": mock_train_loader,
        "mock_val_loader": mock_val_loader,
        "mock_tokenizer": mock_tokenizer,
        "mock_scaler": mock_scaler,
    }

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Creates a temporary directory for checkpoint tests."""
    return tmp_path

# --- Test for Trainer __init__ (Revised Patching & Args) --- #
@patch("craft.training.trainer.CheckpointManager")
@patch("craft.training.trainer.CallbackList")
@patch("hydra.utils.instantiate")
def test_trainer_init(
    mock_hydra_instantiate,
    MockCallbackList,
    MockCheckpointManager,
    setup_trainer_test_environment # Use new fixture
):
    """Test Trainer initialization with new signature."""
    
    mock_checkpoint_manager_instance = MockCheckpointManager.return_value
    
    # Configure hydra mock
    mock_hydra_instantiate.return_value = {'train': MagicMock(), 'val': None} # Mock dataloaders

    # Extract configs from fixture
    training_config = setup_trainer_test_environment["training_config"]
    model_config = setup_trainer_test_environment["model_config_dict"]
    experiment_config = setup_trainer_test_environment["experiment_config"]

    try:
        # Instantiate Trainer with new signature
        trainer = Trainer(
            model_config=model_config,
            config=training_config,
            experiment_config=experiment_config,
            # device=training_config.device, # Get device from config
            # experiment_name="test_init" # Optional experiment name
        )
        
        MockCheckpointManager.assert_called_once() 
        MockCallbackList.assert_not_called() # Should not be called in init
        # Assert internal callback list is None initially
        # assert trainer.callbacks is None # Adjust if CallbackList *is* created in init

        assert trainer.checkpoint_manager is mock_checkpoint_manager_instance 
        
        # --- Remove load_checkpoint assertion --- #
        # load_checkpoint is only called if resume_from_checkpoint is provided
        mock_checkpoint_manager_instance.load_checkpoint.assert_not_called()

    except Exception as e:
        pytest.fail(f"Trainer initialization failed: {e}")

@patch("src.craft.training.trainer.TrainingLoop")
@patch("src.craft.training.trainer.Evaluator")
@patch("src.craft.training.trainer.CheckpointManager")
@patch("src.craft.training.trainer.CallbackList")
@patch("src.craft.training.trainer.ProgressTracker")
@patch("hydra.utils.instantiate")
def test_trainer_train_flow(
    mock_hydra_instantiate,
    MockProgressTracker, MockCallbackList, MockCheckpointManager,
    MockEvaluator, MockTrainingLoop,
    setup_trainer_test_environment, # Use new fixture
    temp_checkpoint_dir
):
    """Test the main training flow coordination within Trainer.train()."""
    # Extract components from the fixture
    training_config = setup_trainer_test_environment["training_config"]
    model_config = setup_trainer_test_environment["model_config_dict"]
    experiment_config = setup_trainer_test_environment["experiment_config"]
    mock_model = setup_trainer_test_environment["mock_model"]
    mock_optimizer = setup_trainer_test_environment["mock_optimizer"]
    mock_train_loader = setup_trainer_test_environment["mock_train_loader"]
    mock_val_loader = setup_trainer_test_environment["mock_val_loader"]
    mock_tokenizer = setup_trainer_test_environment["mock_tokenizer"]
    mock_scaler = setup_trainer_test_environment["mock_scaler"]

    # Configure side effect for hydra instantiate
    # 1st call (dataloaders): returns dict with train/val loaders
    # 2nd call (optimizer): returns the mock optimizer
    # 3rd call (scheduler): returns None (or a mock scheduler if needed)
    mock_hydra_instantiate.side_effect = [
        {'train': mock_train_loader, 'val': mock_val_loader}, # Dataloaders
        mock_optimizer, # Optimizer
        None # Scheduler (assuming no scheduler in this minimal config)
    ]

    # Mocks for internal components created by Trainer
    mock_training_loop_instance = MockTrainingLoop.return_value
    mock_evaluator_instance = MockEvaluator.return_value
    mock_checkpoint_manager_instance = MockCheckpointManager.return_value
    mock_callbacks_instance = MockCallbackList.return_value
    mock_progress_tracker_instance = MockProgressTracker.return_value

    # Mock return values and behaviors
    steps_per_epoch = len(mock_train_loader)
    total_steps = training_config.num_epochs * steps_per_epoch
    mock_evaluator_instance.evaluate.return_value = {"loss": 0.5} # Example eval result
    # Simulate train_epoch advancing state
    def train_epoch_side_effect(*args, **kwargs):
        current_epoch = kwargs.get("current_epoch", 0)
        global_step = kwargs.get("global_step", 0)
        final_step = global_step + steps_per_epoch
        # Simulate loss reduction
        loss = 0.9 - (current_epoch * 0.1)
        return {"final_global_step": final_step, "loss": loss}
    mock_training_loop_instance.train_epoch.side_effect = train_epoch_side_effect

    # Instantiate Trainer (using patched hydra calls internally)
    trainer = Trainer(
        model_config=model_config,
        config=training_config,
        experiment_config=experiment_config,
    )

    # Call the train method
    trainer.train()

    # --- Assertions ---
    # Check hydra instantiate calls
    assert mock_hydra_instantiate.call_count == 3 # Dataloaders, Optimizer, Scheduler
    # Could add more specific checks on hydra call args if needed

    # Check ProgressTracker initialization
    MockProgressTracker.assert_called_once_with(
        total_epochs=training_config.num_epochs,
        steps_per_epoch=steps_per_epoch,
        resume_epoch=0, # Starting from scratch
        resume_step_in_epoch=0
    )

    # Check CallbackList initialization and calls
    MockCallbackList.assert_called_once_with(ANY, trainer=trainer) # Callbacks are instantiated
    mock_callbacks_instance.on_train_begin.assert_called_once()
    assert mock_callbacks_instance.on_epoch_begin.call_count == training_config.num_epochs
    assert mock_callbacks_instance.on_epoch_end.call_count == training_config.num_epochs
    mock_callbacks_instance.on_train_end.assert_called_once()

    # Check TrainingLoop initialization and calls
    MockTrainingLoop.assert_called_once_with(
        model=mock_model, # Should use the model passed to Trainer -> setup_components
        optimizer=mock_optimizer, # Should use the optimizer created by hydra
        criterion=ANY, # Trainer likely creates this internally
        device=ANY, # Trainer detects/sets device
        scaler=ANY, # Trainer creates scaler
        progress_tracker=mock_progress_tracker_instance,
        callbacks=mock_callbacks_instance,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps
    )
    assert mock_training_loop_instance.train_epoch.call_count == training_config.num_epochs
    # Verify args passed to train_epoch for the first epoch
    mock_training_loop_instance.train_epoch.assert_any_call(
        dataloader=mock_train_loader, current_epoch=0, global_step=0
    )

    # Check Evaluator initialization and calls (if eval happens)
    if training_config.eval_interval and mock_val_loader:
        MockEvaluator.assert_called_once_with(
            model=mock_model,
            device=ANY,
            progress_tracker=mock_progress_tracker_instance,
            callbacks=mock_callbacks_instance
        )
        # Eval happens every epoch if eval_interval = 1
        num_evals = training_config.num_epochs // training_config.eval_interval
        assert mock_evaluator_instance.evaluate.call_count == num_evals
        mock_evaluator_instance.evaluate.assert_called_with(dataloader=mock_val_loader)
    else:
        MockEvaluator.assert_not_called()
        mock_evaluator_instance.evaluate.assert_not_called()

    # Check CheckpointManager initialization and calls (if saving happens)
    if training_config.save_interval:
        MockCheckpointManager.assert_called_once() # Instantiated during Trainer init
        num_saves = training_config.num_epochs // training_config.save_interval
        # +1 potentially for save_last, or save_best if eval happened
        # This needs refinement based on actual save logic (save_last, save_best)
        # Simple check for now:
        assert mock_checkpoint_manager_instance.save_checkpoint.call_count >= num_saves

        # Example check for the first save call (end of epoch 0 if save_interval=1)
        if training_config.save_interval == 1:
             expected_state_epoch0 = TrainingState(
                 epoch=0, # Epoch just completed
                 global_step=steps_per_epoch, # Steps completed in epoch 0
                 model_state_dict=ANY,
                 optimizer_state_dict=ANY,
                 scaler_state_dict=ANY,
                 scheduler_state_dict=None, # Assuming no scheduler
                 best_val_metric=ANY # Might be initial value or from first eval
             )
             mock_checkpoint_manager_instance.save_checkpoint.assert_any_call(
                 state=expected_state_epoch0,
                 is_best=ANY, # Depends on eval result
                 tokenizer=mock_tokenizer # If tokenizer exists
             )
    else:
        # If saving disabled, CheckpointManager might still be initialized
        # but save should not be called
        mock_checkpoint_manager_instance.save_checkpoint.assert_not_called()

    # Check final progress state (example)
    assert mock_progress_tracker_instance.current_epoch == training_config.num_epochs
    assert mock_progress_tracker_instance.global_step == total_steps

@patch("src.craft.training.trainer.TrainingLoop")
@patch("src.craft.training.trainer.Evaluator")
@patch("src.craft.training.trainer.CheckpointManager")
@patch("src.craft.training.trainer.CallbackList")
@patch("src.craft.training.trainer.ProgressTracker")
@patch("hydra.utils.instantiate")
def test_trainer_resume_from_checkpoint(
    mock_hydra_instantiate,
    MockProgressTracker, MockCallbackList, MockCheckpointManager,
    MockEvaluator, MockTrainingLoop,
    setup_trainer_test_environment, # Use new fixture
    temp_checkpoint_dir
):
    """Test Trainer correctly resumes state using internally created components."""
    # Extract components from the fixture
    training_config = setup_trainer_test_environment["training_config"]
    model_config = setup_trainer_test_environment["model_config_dict"]
    experiment_config = setup_trainer_test_environment["experiment_config"]
    mock_model = setup_trainer_test_environment["mock_model"]
    mock_optimizer = setup_trainer_test_environment["mock_optimizer"]
    mock_train_loader = setup_trainer_test_environment["mock_train_loader"]
    mock_val_loader = setup_trainer_test_environment["mock_val_loader"]
    mock_tokenizer = setup_trainer_test_environment["mock_tokenizer"]
    mock_scaler = setup_trainer_test_environment["mock_scaler"]

    # Modify config for this test
    config: TrainingConfig = training_config # Use extracted config object
    resume_path = str(temp_checkpoint_dir / "dummy_ckpt.pt")
    config_updates = {
        "resume_from_checkpoint": resume_path,
        "num_epochs": 3, # Total epochs for the run
        "eval_interval": 1,
        "save_interval": 1
    }
    config = config.model_copy(update=config_updates)

    # Update experiment_config node to reflect resume path (Trainer reads from here)
    experiment_config_copy = experiment_config.copy() # Don't modify fixture directly
    if not hasattr(experiment_config_copy, 'checkpoints'):
        experiment_config_copy.checkpoints = OmegaConf.create()
    experiment_config_copy.checkpoints.resume_from_checkpoint = resume_path

    # --- Configure Mocks ---
    # Configure side effect for hydra instantiate (critical for resume)
    # Needs to provide dataloaders, optimizer, scheduler
    mock_hydra_instantiate.side_effect = [
        {'train': mock_train_loader, 'val': mock_val_loader}, # Dataloaders
        mock_optimizer, # Optimizer
        None # Scheduler
    ]

    mock_training_loop_instance = MockTrainingLoop.return_value
    mock_evaluator_instance = MockEvaluator.return_value
    mock_checkpoint_manager_instance = MockCheckpointManager.return_value
    mock_callbacks_instance = MockCallbackList.return_value
    mock_progress_tracker_instance = MockProgressTracker.return_value

    # Mock CheckpointManager behavior for loading
    mock_checkpoint_manager_instance.checkpoint_dir = str(temp_checkpoint_dir)
    steps_per_epoch = len(mock_train_loader)
    loaded_epoch = 1 # Resume from end of epoch 1
    loaded_global_step = steps_per_epoch * (loaded_epoch + 1) # Global step *after* epoch 1
    loaded_best_val = 0.6
    # Simulate state dicts - can be simple mocks/empty dicts if not strictly checked
    loaded_model_state = {'param1': torch.tensor(1.0)}
    loaded_optimizer_state = {'state': {}, 'param_groups': []}
    loaded_scaler_state = mock_scaler.state_dict() # Use mock scaler's state
    loaded_state = TrainingState(
        epoch=loaded_epoch,
        global_step=loaded_global_step,
        model_state_dict=loaded_model_state,
        optimizer_state_dict=loaded_optimizer_state,
        scaler_state_dict=loaded_scaler_state, # Include scaler state
        best_val_metric=loaded_best_val
        # scheduler_state_dict = None # Assuming no scheduler state
    )
    mock_checkpoint_manager_instance.load_checkpoint.return_value = loaded_state

    # Mock evaluate and train_epoch behavior for remaining epochs
    mock_evaluator_instance.evaluate.return_value = {"loss": 0.55}
    def train_epoch_side_effect(*args, **kwargs):
        current_epoch = kwargs.get("current_epoch", 0)
        global_step = kwargs.get("global_step", 0)
        final_step = global_step + steps_per_epoch
        return {"final_global_step": final_step, "loss": 0.2}
    mock_training_loop_instance.train_epoch.side_effect = train_epoch_side_effect

    # --- Instantiate Trainer ---
    # Trainer should use the resume path from config/experiment_config
    trainer = Trainer(
        model_config=model_config,
        config=config, # Use the updated config for this test
        experiment_config=experiment_config_copy, # Use the copy with resume path
    )

    # --- Call train ---
    trainer.train()

    # --- Assertions ---
    # 1. Check hydra calls (still 3 expected: dataloaders, optim, sched)
    assert mock_hydra_instantiate.call_count == 3

    # 2. Check CheckpointManager load call
    mock_checkpoint_manager_instance.load_checkpoint.assert_called_once_with(resume_path=resume_path, tokenizer=mock_tokenizer, device=ANY)
    # Verify state was loaded into components (use ANY for simplicity or specific values)
    mock_model.load_state_dict.assert_called_once_with(loaded_model_state)
    mock_optimizer.load_state_dict.assert_called_once_with(loaded_optimizer_state)
    mock_scaler.load_state_dict.assert_called_once_with(loaded_scaler_state)

    # 3. Check ProgressTracker initialization with resumed state
    MockProgressTracker.assert_called_once_with(
        total_epochs=config.num_epochs,
        steps_per_epoch=steps_per_epoch,
        resume_epoch=loaded_epoch + 1, # Should start *after* the loaded epoch
        resume_step_in_epoch=0 # Assuming resume happens between epochs
    )

    # 4. Check TrainingLoop calls for the *remaining* epochs
    remaining_epochs = config.num_epochs - (loaded_epoch + 1)
    assert mock_training_loop_instance.train_epoch.call_count == remaining_epochs
    # Check first resumed epoch call (epoch 2, starting from global_step loaded)
    mock_training_loop_instance.train_epoch.assert_any_call(
        dataloader=mock_train_loader, current_epoch=loaded_epoch + 1, global_step=loaded_global_step
    )

    # 5. Check Evaluator calls for remaining epochs
    # Eval happens every epoch (interval=1), starting from the resumed epoch
    num_evals_remaining = remaining_epochs
    assert mock_evaluator_instance.evaluate.call_count == num_evals_remaining

    # 6. Check Checkpoint saving for remaining epochs
    num_saves_remaining = remaining_epochs
    # +1 possibly for save_last or final best
    # Check that save was called *after* loading
    assert mock_checkpoint_manager_instance.save_checkpoint.call_count >= num_saves_remaining

    # 7. Check Callback calls (ensure they span the whole resumed run)
    mock_callbacks_instance.on_train_begin.assert_called_once()
    assert mock_callbacks_instance.on_epoch_begin.call_count == remaining_epochs
    assert mock_callbacks_instance.on_epoch_end.call_count == remaining_epochs
    mock_callbacks_instance.on_train_end.assert_called_once()

    # Check callbacks received loaded state info (e.g., on_train_begin)
    mock_callbacks_instance.on_train_begin.assert_called_once_with(logs={'resumed_from': resume_path})

# Test exception handling (Optional example, might need adjustment)
# @patch("src.craft.training.trainer.TrainingLoop")
# ... other patches
# def test_trainer_handles_training_exception(...)
# ... setup ...
# mock_training_loop_instance.train_epoch.side_effect = Exception("Training failed!")
# with pytest.raises(Exception, match="Training failed!"):
#     trainer.train()
# mock_callbacks_instance.on_train_end.assert_called_once_with(exception=ANY) # Check exception passed

# --- More Tests (If any) --- # 