import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch, ANY, call
import logging
import tempfile
import copy
import types
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError
from craft.config.schemas import TrainingConfig, LanguageModelConfig
from craft.models.base import LanguageModel
from craft.training.callbacks import Callback, CallbackList
from craft.training.evaluation import Evaluator
from craft.training.checkpointing import CheckpointManager, TrainingState
from craft.training.progress import ProgressTracker
from craft.training.training_loop import TrainingLoop
from craft.training.trainer import Trainer, initialize_callbacks
from craft.data.base import BaseDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from craft.training.callbacks.sample_generation import SampleGenerationCallback
from craft.data.tokenizers.base import Tokenizer

# --- Mocks and Fixtures --- #

# Minimal Model Config Fixture (similar to other files)
@pytest.fixture
def minimal_model_config_dict():
    # Define a simple mock model config
    # return {'_target_': 'unittest.mock.MagicMock', 'architecture': 'mock'}
    # FIXED: Add nested 'config' block as expected by initialize_model
    return {
        '_target_': 'unittest.mock.MagicMock',
        'config': { # Nested block
            'architecture': 'mock'
            # Add other dummy params if needed by model init/validation
        }
    }

# --- Tests for Trainer --- #

@pytest.fixture
def setup_trainer_test_environment(minimal_model_config_dict):
    """Provides configuration and component mocks needed for Trainer tests."""
    # --- Create TrainingConfig object --- #
    training_args = {
        # _target_ is implicitly handled by Trainer init now
        "batch_size": 4,
        "num_epochs": 2,
        "use_amp": False,
        "gradient_accumulation_steps": 1,
        "log_interval": 10,
        "eval_interval": 50,
        "save_interval": 100,
        "keep_last": 2,
        "resume_from_checkpoint": None,
        "log_level": "DEBUG",
        "seed": 42,
        "device": "cpu",
        "max_steps": None,
        "learning_rate": 1e-4,
        "max_grad_norm": 1.0,
        "torch_compile": False, # Likely handled separately
        "sample_max_new_tokens": 100,
        "sample_temperature": 0.8,
        "sample_start_text": "Once upon a time",
        "val_metric": "loss",
        "time_save_interval_seconds": 0,
        "time_eval_interval_seconds": 0,
        "mixed_precision": False, # Potentially redundant with use_amp
        "save_steps_interval": 0,
        "compile_model": False, # Changed from torch_compile?
    }
    try:
        pydantic_config = TrainingConfig(**training_args)
    except ValidationError as e:
        pytest.fail(f"Failed to create valid TrainingConfig in fixture: {e}")

    # --- Mock components --- #
    mock_model = MagicMock(spec=LanguageModel)
    mock_model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.randn(1))]))
    mock_model.config = MagicMock()
    mock_model.config.vocab_size = 50
    mock_optimizer = MagicMock(spec=AdamW)
    mock_train_dataset = MagicMock(spec=BaseDataset)
    mock_train_dataset.__len__.return_value = 100
    mock_val_dataset = MagicMock(spec=BaseDataset)
    mock_val_dataset.__len__.return_value = 20
    mock_train_loader = DataLoader(mock_train_dataset, batch_size=training_args['batch_size'])
    mock_val_loader = DataLoader(mock_val_dataset, batch_size=training_args['batch_size'])
    mock_tokenizer = MagicMock(spec=Tokenizer)
    mock_tokenizer.get_vocab_size.return_value = 50257 # Example vocab size
    mock_tokenizer.vocab_size = 50
    mock_scaler = MagicMock(spec=torch.cuda.amp.GradScaler)
    mock_scaler.state_dict.return_value = {}
    mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
    mock_checkpoint_manager = MagicMock(spec=CheckpointManager)
    mock_evaluator = MagicMock(spec=Evaluator)
    mock_tensorboard_logger = MagicMock(spec=SummaryWriter)
    mock_sample_generator = MagicMock(spec=SampleGenerationCallback)
    mock_callback_list = MagicMock(spec=CallbackList)
    # Create mock callbacks explicitly for easier assertion later
    mock_callbacks_instance = [
        mock_tensorboard_logger,
        mock_sample_generator,
        # Add other standard mocks if needed
    ]
    mock_callback_list.callbacks = mock_callbacks_instance


    # --- Create Experiment Config Node (Mimicking Hydra Structure) --- #
    experiment_inner_dict = { # RENAME to reflect nesting
        'name': 'test_experiment',
        'output_dir': 'outputs/test_experiment',
        'device': training_args['device'],
        'training': OmegaConf.create(training_args),
        'model': OmegaConf.create(minimal_model_config_dict),
        'data': OmegaConf.create({
            'tokenizer': { '_target_': 'unittest.mock.MagicMock' },
            'batch_size': training_args['batch_size'],
            'num_workers': 0,
             'datasets': { # FIXED: Nest dataset config under 'dataset' key
                'train': {
                    'dataset': { '_target_': 'unittest.mock.MagicMock' }
                    # Add dataloader config here if needed
                },
                'val': {
                     'dataset': { '_target_': 'unittest.mock.MagicMock' }
                },
                'test': None
                }
        }),
        'optimizer': OmegaConf.create({
            '_target_': 'unittest.mock.MagicMock',
            'lr': training_args['learning_rate']
            }),
        'scheduler': None,
        'validation': OmegaConf.create({
            'enable': False, # Default to False, test might override
            'val_interval': 1 # Default interval
        }),
        'callbacks': { # Mock callback configs
             'tensorboard_logger': {'_target_': 'path.to.TensorBoardLogger'},
             'sample_generation': {'_target_': 'path.to.SampleGenerationCallback'},
        },
        'checkpointing': OmegaConf.create({
            '_target_': 'craft.training.checkpointing.CheckpointManager',
            'checkpoint_dir': './pytest_trainer_cm_dir',
            'experiment_name': 'test_fixture_exp',
            'keep_last_n': training_args['keep_last'],
        }),
        'eval': OmegaConf.create({
            '_target_': 'craft.training.evaluation.Evaluator',
            'config': {}
        }),
    }
    experiment_conf_dict = {
         'experiment': experiment_inner_dict
    }
    experiment_config_node = OmegaConf.create(experiment_conf_dict)

    return {
        "experiment_config": experiment_config_node,
        "expected_training_config": pydantic_config,
        # Component Mocks (to be returned by patched hydra.utils.instantiate)
        "mock_model": mock_model,
        "mock_optimizer": mock_optimizer,
        "mock_train_loader": mock_train_loader,
        "mock_val_loader": mock_val_loader,
        "mock_tokenizer": mock_tokenizer,
        "mock_scaler": mock_scaler, # Scaler might be created internally based on use_amp
        "mock_scheduler": mock_scheduler,
        "mock_checkpoint_manager": mock_checkpoint_manager,
        "mock_evaluator": mock_evaluator,
        "mock_tensorboard_logger": mock_tensorboard_logger,
        "mock_sample_generator": mock_sample_generator,
        "mock_callback_list": mock_callback_list, # Mock for the list itself
    }

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Creates a temporary directory for checkpoint tests."""
    return tmp_path

# --- Test for Trainer __init__ (Refactored) --- #
@patch("craft.training.trainer.torch.cuda.amp.GradScaler") # Keep for now, though scaler init is complex
@patch("craft.training.trainer.CallbackList") # Keep to check it's called
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
# REMOVED @patch("hydra.utils.instantiate")
def test_trainer_init_refactored(
    # Pass mocks in reverse order of decorators
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
    MockCallbackList, # The mocked Class
    MockGradScaler, # The mocked Class (maybe remove?)
    setup_trainer_test_environment
):
    """Test Trainer initialization using patched initialization helpers."""
    fixture_data = setup_trainer_test_environment
    experiment_config = fixture_data["experiment_config"]
    expected_training_config = fixture_data["expected_training_config"]
    exp_cfg = experiment_config.experiment # Shortcut

    # --- Configure mocks for patched initializers --- #
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_tokenizer.return_value = fixture_data["mock_tokenizer"]
    mock_initialize_model.return_value = fixture_data["mock_model"]
    mock_initialize_dataloaders.return_value = (
        fixture_data["mock_train_loader"],
        fixture_data["mock_val_loader"]
    )
    mock_initialize_optimizer.return_value = fixture_data["mock_optimizer"]
    mock_initialize_scheduler.return_value = fixture_data["mock_scheduler"] # Will be None if scheduler cfg is None
    mock_initialize_amp_scaler.return_value = fixture_data["mock_scaler"]
    # initialize_callbacks returns a LIST of raw callback instances
    mock_raw_callback_list = [
        fixture_data["mock_tensorboard_logger"],
        fixture_data["mock_sample_generator"]
    ]
    mock_initialize_callbacks.return_value = mock_raw_callback_list
    mock_initialize_checkpoint_manager.return_value = fixture_data["mock_checkpoint_manager"]
    # Handle case where val_loader might be None -> evaluator is None
    mock_evaluator_instance = fixture_data["mock_evaluator"] if fixture_data["mock_val_loader"] else None
    mock_initialize_evaluator.return_value = mock_evaluator_instance
    # Mock compile_model to just return the model passed to it (no-op)
    mock_compile_model.side_effect = lambda model, *args, **kwargs: model

    # Configure CallbackList mock (the class patch)
    # It should be called with the list from initialize_callbacks
    # We also need an instance for trainer.callbacks assertion
    mock_callback_list_instance = MagicMock(spec=CallbackList)
    mock_callback_list_instance.callbacks = mock_raw_callback_list # Simulate internal list
    MockCallbackList.return_value = mock_callback_list_instance

    # --- Instantiate Trainer (will use patched initializers) --- #
    trainer = Trainer(cfg=experiment_config)

    # --- Assertions --- #
    # 1. Direct attributes
    assert isinstance(trainer.config, TrainingConfig)
    assert trainer.config == expected_training_config
    assert trainer.device == torch.device("cpu") # Check against value returned by mock_initialize_device

    # 2. Instantiated components (check they are the mocks returned by initializers)
    assert trainer.model is fixture_data["mock_model"]
    assert trainer.optimizer is fixture_data["mock_optimizer"]
    assert trainer.train_dataloader is fixture_data["mock_train_loader"]
    assert trainer.val_dataloader is fixture_data["mock_val_loader"]
    assert trainer.tokenizer is fixture_data["mock_tokenizer"]
    assert trainer.scheduler is mock_initialize_scheduler.return_value # Check correct assignment
    assert trainer.checkpoint_manager is fixture_data["mock_checkpoint_manager"]
    assert trainer.evaluator is mock_evaluator_instance # Check correct assignment
    assert trainer.callbacks is mock_callback_list_instance # Check the LIST instance
    assert trainer.scaler is fixture_data["mock_scaler"]

    # 3. Verify calls to initialization functions
    mock_initialize_device.assert_called_once_with(exp_cfg.get("device", "cpu"))
    mock_initialize_tokenizer.assert_called_once_with(exp_cfg.get("data"))
    # initialize_model gets the *uncompiled* model instance first
    # The mock returns fixture_data["mock_model"]. Check args passed to it.
    mock_initialize_model.assert_called_once_with(
        exp_cfg.get("model"), # model config node
        mock_initialize_device.return_value, # device
        mock_initialize_tokenizer.return_value # tokenizer
    )
    mock_initialize_dataloaders.assert_called_once_with(
        exp_cfg.get("data"), # data config node
        mock_initialize_device.return_value, # device
        mock_initialize_tokenizer.return_value # tokenizer
    )
    # initialize_optimizer receives the model returned by initialize_model
    mock_initialize_optimizer.assert_called_once_with(exp_cfg.get("optimizer"), fixture_data["mock_model"])
    mock_initialize_scheduler.assert_called_once_with(exp_cfg.get("scheduler"), fixture_data["mock_optimizer"])
    mock_initialize_amp_scaler.assert_called_once_with(trainer.config.use_amp, mock_initialize_device.return_value)
    mock_initialize_callbacks.assert_called_once_with(exp_cfg.get("callbacks"))
    # Check CallbackList class was called with the result of initialize_callbacks
    MockCallbackList.assert_called_once_with(mock_raw_callback_list)
    # Check initialize_checkpoint_manager call carefully
    mock_initialize_checkpoint_manager.assert_called_once_with(
        checkpoint_cfg_node=exp_cfg.get("checkpointing"),
        full_app_config=experiment_config,
        experiment_name=trainer.experiment_name,
        model=fixture_data["mock_model"], # Model instance from initialize_model mock
        optimizer=fixture_data["mock_optimizer"],
        scaler=fixture_data["mock_scaler"],
        scheduler=mock_initialize_scheduler.return_value,
        callbacks=mock_callback_list_instance, # The CallbackList instance
        tokenizer=fixture_data["mock_tokenizer"],
    )
    mock_initialize_evaluator.assert_called_once_with(
        eval_cfg_node=exp_cfg.get("evaluation"),
        model=fixture_data["mock_model"],
        val_dataloader=fixture_data["mock_val_loader"],
        device=mock_initialize_device.return_value,
        use_amp=trainer.config.use_amp,
        callbacks=mock_callback_list_instance,
    )
    compile_options_cfg = experiment_config.get("torch_compile_options")
    mock_compile_model.assert_called_once_with(
        fixture_data["mock_model"], # Model instance before compilation attempt
        trainer.compile_model,
        compile_options_cfg
    )

    # 4. Checkpoint loading should not happen on basic init
    if trainer.checkpoint_manager:
        # Access the mock directly returned by the initializer patch
        mock_initialize_checkpoint_manager.return_value.load_checkpoint.assert_not_called()

@patch("craft.training.trainer.TrainingLoop")
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
# REMOVED @patch("hydra.utils.instantiate")
@patch("craft.training.trainer.torch.cuda.amp.GradScaler") # Keep?
@patch("craft.training.trainer.CallbackList") # Keep
def test_trainer_train_flow_refactored(
    MockCallbackList, # The class
    MockGradScaler, # The class
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
    MockTrainingLoop, # The class
    setup_trainer_test_environment,
    temp_checkpoint_dir
):
    """Test the main training flow using patched initialization helpers."""
    fixture_data = setup_trainer_test_environment
    experiment_config = fixture_data["experiment_config"]
    expected_training_config = fixture_data["expected_training_config"]

    # --- Configure mocks for patched initializers --- #
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_tokenizer.return_value = fixture_data["mock_tokenizer"]
    mock_initialize_model.return_value = fixture_data["mock_model"]
    mock_initialize_dataloaders.return_value = (
        fixture_data["mock_train_loader"],
        fixture_data["mock_val_loader"]
    )
    mock_initialize_optimizer.return_value = fixture_data["mock_optimizer"]
    mock_initialize_scheduler.return_value = fixture_data["mock_scheduler"]
    mock_initialize_amp_scaler.return_value = fixture_data["mock_scaler"]
    mock_raw_callback_list = [
        fixture_data["mock_tensorboard_logger"],
        fixture_data["mock_sample_generator"]
    ]
    mock_initialize_callbacks.return_value = mock_raw_callback_list
    # Return the *same* CM instance so we can assert on it later if needed
    mock_checkpoint_manager_instance = fixture_data["mock_checkpoint_manager"]
    mock_initialize_checkpoint_manager.return_value = mock_checkpoint_manager_instance
    mock_evaluator_instance = fixture_data["mock_evaluator"] if fixture_data["mock_val_loader"] else None
    mock_initialize_evaluator.return_value = mock_evaluator_instance
    mock_compile_model.side_effect = lambda model, *args, **kwargs: model

    # --- Configure mocks for Patched Classes --- #
    # TrainingLoop instance mock (returned when Trainer calls MockTrainingLoop(...))
    mock_training_loop_instance = MockTrainingLoop.return_value
    # CallbackList instance mock (returned when Trainer calls MockCallbackList(...))
    mock_callback_list_instance = MagicMock(spec=CallbackList)
    mock_callback_list_instance.callbacks = mock_raw_callback_list
    # Add methods expected by Trainer.train()
    mock_callback_list_instance.on_train_begin = MagicMock()
    mock_callback_list_instance.on_epoch_begin = MagicMock()
    mock_callback_list_instance.on_validation_begin = MagicMock()
    mock_callback_list_instance.on_validation_end = MagicMock()
    mock_callback_list_instance.on_epoch_end = MagicMock()
    mock_callback_list_instance.on_train_end = MagicMock()
    MockCallbackList.return_value = mock_callback_list_instance # Trainer uses this

    # --- Configure Mock Return Values --- #
    if mock_evaluator_instance:
        mock_evaluator_instance.evaluate.return_value = {"loss": 0.5}
    # Simulate TrainingLoop.run returning the final state
    final_step = len(fixture_data["mock_train_loader"]) * expected_training_config.num_epochs
    run_return_dict = {
        "final_global_step": final_step,
        "loss": 0.1,
        "epoch": expected_training_config.num_epochs - 1
    }
    # Configure the .run method on the instance returned by MockTrainingLoop
    # Revert back to returning the dictionary directly
    MockTrainingLoop.return_value.run.return_value = run_return_dict

    # --- Instantiate Trainer --- #
    trainer = Trainer(cfg=experiment_config)

    # --- Call train --- #
    trainer.train()

    # --- Assertions (Focus on Orchestration) --- #

    # 1. Verify TrainingLoop was instantiated correctly
    # Check it was called with the components returned by the patched initializers
    MockTrainingLoop.assert_called_once_with(
        model=fixture_data["mock_model"], # Model instance from initializer
        optimizer=fixture_data["mock_optimizer"], # Optimizer instance from initializer
        train_dataloader=fixture_data["mock_train_loader"],
        device=mock_initialize_device.return_value,
        config=trainer.config, # Pass the TrainingConfig object directly
        scheduler=fixture_data["mock_scheduler"], # Scheduler instance from initializer
        callbacks=mock_callback_list_instance, # The CallbackList *instance*
        checkpoint_manager=mock_checkpoint_manager_instance # ADD THIS ARGUMENT
    )

    # 2. Verify TrainingLoop.run() was called
    mock_training_loop_instance.run.assert_called_once()

    # 3. Verify Callbacks orchestration
    mock_callback_list_instance.on_train_begin.assert_called_once_with(global_step=0)
    # Add checks for epoch begin/end, validation begin/end based on intervals
    # For simplicity, just check train_end for now
    mock_callback_list_instance.on_train_end.assert_called_once_with(metrics=ANY)

    # 4. Verify Evaluator orchestration
    # if trainer.val_dataloader and expected_training_config.eval_interval > 0:
        # Assert evaluate was called (exact number depends on loop logic which is mocked)
        # mock_evaluator_instance.evaluate.assert_called() # COMMENTED OUT: Trainer delegates eval to mocked TrainingLoop.run

    # 5. CheckpointManager - verify saves if intervals dictate (tricky with mocked loop)
    # If save_interval is 100, and loop runs 50 steps (2 epochs * 25 steps/epoch),
    # save_checkpoint should NOT be called by the loop.
    # Let's check based on the mocked loop run result.
    # Since run is mocked, internal saves won't happen. Assert not called.
    if mock_checkpoint_manager_instance:
        mock_checkpoint_manager_instance.save_checkpoint.assert_not_called() # Assuming loop handles saves

    # 6. Model Compilation (check compile_model_if_enabled was called during init)
    # This check belongs more in the init test, but we can ensure it was called once.
    mock_compile_model.assert_called_once()

@patch("craft.training.trainer.initialize_device")
@patch("craft.training.trainer.initialize_model")
@patch("craft.training.trainer.initialize_optimizer")
@patch("craft.training.trainer.initialize_scheduler")
@patch("craft.training.trainer.initialize_dataloaders")
@patch("craft.training.trainer.initialize_callbacks")
@patch("craft.training.trainer.initialize_checkpoint_manager")
@patch("craft.training.trainer.initialize_evaluator")
@patch("craft.training.trainer.compile_model_if_enabled")
@patch("craft.training.trainer.TrainingLoop")
@patch("craft.training.trainer.CallbackList")
def test_trainer_resume_from_checkpoint_refactored(
    # Corrected parameter order and types:
    MockCallbackList: MagicMock, # from @patch("...CallbackList")
    MockTrainingLoop: MagicMock, # from @patch("...TrainingLoop")
    mock_compile_model_if_enabled: MagicMock, # from @patch("...compile_model...")
    mock_initialize_evaluator: MagicMock, # from @patch("...initialize_evaluator")
    mock_initialize_checkpoint_manager: MagicMock, # from @patch("...initialize_checkpoint_manager")
    mock_initialize_callbacks: MagicMock, # from @patch("...initialize_callbacks")
    mock_initialize_dataloaders: MagicMock, # from @patch("...initialize_dataloaders")
    mock_initialize_scheduler: MagicMock, # from @patch("...initialize_scheduler")
    mock_initialize_optimizer: MagicMock, # from @patch("...initialize_optimizer")
    mock_initialize_model: MagicMock, # from @patch("...initialize_model")
    mock_initialize_device: MagicMock, # from @patch("...initialize_device")
    # Remaining parameters:
    setup_trainer_test_environment, # Use the fixture directly
    tmp_path: Path,
):
    """Tests Trainer initialization and training flow when resuming from a checkpoint."""
    fixture_data = setup_trainer_test_environment # Assign fixture result locally
    # --- Setup: Experiment Config & Data --- #
    experiment_config_copy = copy.deepcopy(fixture_data["experiment_config"])
    total_epochs = 3 # Define total epochs for the test scenario
    experiment_config_copy.experiment.training.num_epochs = total_epochs
    # Ensure validation is enabled if mock_val_loader is provided
    if fixture_data["mock_val_loader"] is not None:
        experiment_config_copy.experiment.validation.enable = True
        experiment_config_copy.experiment.validation.val_interval = 1 # Validate every epoch
    else:
        experiment_config_copy.experiment.validation.enable = False

    resume_path = tmp_path / "model.ckpt"
    resume_path.touch() # Create a dummy checkpoint file

    # --- Setup: Expected State After Loading Checkpoint --- #
    # This is the state we *expect* the CheckpointManager's load_checkpoint to restore
    expected_resumed_epoch = 1
    expected_resumed_global_step = len(fixture_data["mock_train_loader"]) # Steps per epoch
    expected_resumed_best_val_metric = 0.6
    expected_resumed_state_dict = { # Rename to indicate it's a dict
        "epoch": expected_resumed_epoch,
        "global_step": expected_resumed_global_step,
        "best_val_metric": expected_resumed_best_val_metric,
        "model_state_dict": {"param": torch.tensor(1.0)}, # Dummy state
        "optimizer_state_dict": {"param_groups": []}, # Dummy state
        "scheduler_state_dict": {"last_epoch": expected_resumed_epoch}, # Dummy state
        "rng_state": {"cpu_rng_state": torch.get_rng_state()}, # Dummy state
        "scaler_state_dict": None # Assuming no AMP initially for simplicity
    }
    # Wrap the dict in a SimpleNamespace to allow attribute access
    expected_resumed_state_obj = types.SimpleNamespace(**expected_resumed_state_dict)

    # --- Configure Mock Return Values for Initialization --- #
    # Mocks for component initializers
    mock_initialize_device.return_value = torch.device("cpu")
    mock_initialize_model.return_value = fixture_data["mock_model"]
    mock_initialize_optimizer.return_value = fixture_data["mock_optimizer"]
    mock_initialize_scheduler.return_value = fixture_data["mock_scheduler"]
    # Configure the dataloader mock
    mock_initialize_dataloaders.return_value = (
        fixture_data["mock_train_loader"],
        fixture_data["mock_val_loader"],
    )
    # Mock initialize_callbacks to return the *list* of raw callbacks
    mock_raw_callback_list = [
        MagicMock(spec=Callback) for _ in range(2) # Create dummy raw callbacks
    ]
    mock_initialize_callbacks.return_value = mock_raw_callback_list
    # IMPORTANT: initialize_checkpoint_manager returns the *instance* whose load_checkpoint method is patched
    mock_checkpoint_manager_instance = fixture_data["mock_checkpoint_manager"]
    mock_initialize_checkpoint_manager.return_value = mock_checkpoint_manager_instance
    # Configure the *instance*'s load_checkpoint method
    mock_checkpoint_manager_instance.load_checkpoint.return_value = expected_resumed_state_obj # Return the object

    mock_evaluator_instance = fixture_data["mock_evaluator"] if fixture_data["mock_val_loader"] else None
    mock_initialize_evaluator.return_value = mock_evaluator_instance
    mock_compile_model_if_enabled.side_effect = lambda model, *args, **kwargs: model

    # --- Instantiate Trainer --- #
    trainer = Trainer(
        cfg=experiment_config_copy, # Pass the modified experiment config
        resume_from_checkpoint=resume_path
    )

    # --- Assertions During/After Initialization --- #
    # 1. Check CheckpointManager.load_checkpoint was called ONCE during init
    mock_checkpoint_manager_instance.load_checkpoint.assert_called_once_with(resume_path)

    # 2. Check initialize_X calls (should happen AFTER load_checkpoint)
    mock_initialize_model.assert_called_once()
    mock_initialize_optimizer.assert_called_once()
    mock_initialize_scheduler.assert_called_once()
    mock_initialize_dataloaders.assert_called_once() # Add back check for dataloader init call
    mock_initialize_callbacks.assert_called_once()
    mock_initialize_checkpoint_manager.assert_called_once()
    mock_initialize_evaluator.assert_called_once()
    mock_compile_model_if_enabled.assert_called_once()

    # 3. Check Trainer internal state is correctly restored *from loaded state*
    assert trainer.epoch == expected_resumed_epoch
    assert trainer.global_step == expected_resumed_global_step
    assert trainer.best_val_metric == expected_resumed_best_val_metric
    # Check dataloader assignment on trainer instance
    assert trainer.train_dataloader is fixture_data["mock_train_loader"]
    assert trainer.val_dataloader is fixture_data["mock_val_loader"]

    # 4. Check CallbackList instantiation and setup
    MockCallbackList.assert_called_once_with(mock_raw_callback_list)
    assert trainer.callbacks is MockCallbackList.return_value # Check the list instance


    # --- Call train --- #
    # Configure mocks for Patched Classes needed during train()
    mock_training_loop_instance = MockTrainingLoop.return_value # Get the instance mock
    mock_callback_list_instance = MockCallbackList.return_value # Get the instance mock (re-fetch for clarity)
    # Add methods expected by Trainer.train() to the CallbackList *instance* mock
    mock_callback_list_instance.on_train_begin = MagicMock()
    mock_callback_list_instance.on_train_end = MagicMock()

    # Configure Mock Return Values for Training Flow (needed by train() assertions)
    # We are resuming from step `expected_resumed_global_step`
    final_step_in_run = len(fixture_data["mock_train_loader"]) * (total_epochs - expected_resumed_epoch)
    final_global_step_overall = expected_resumed_global_step + final_step_in_run
    # Configure the .run method on the TrainingLoop *instance* mock
    run_return_dict = {
        "final_global_step": final_global_step_overall, # Final step reached in this run
        "loss": 0.2,
        "epoch": total_epochs - 1 # Final epoch completed
    }
    mock_training_loop_instance.run.return_value = run_return_dict

    training_result = trainer.train()

    # --- Assertions after train() --- #
    # 1. Check TrainingLoop instantiation
    MockTrainingLoop.assert_called_once_with(
        model=fixture_data["mock_model"], # Model instance from initializer
        optimizer=fixture_data["mock_optimizer"], # Optimizer instance from initializer
        train_dataloader=fixture_data["mock_train_loader"],
        device=mock_initialize_device.return_value,
        config=trainer.config, # Pass the TrainingConfig object directly
        scheduler=fixture_data["mock_scheduler"], # Scheduler instance from initializer
        callbacks=mock_callback_list_instance, # The CallbackList *instance*
        checkpoint_manager=mock_checkpoint_manager_instance # ADD THIS ARGUMENT
    )

    # 2. Check TrainingLoop.run call with correct *starting* state
    mock_training_loop_instance.run.assert_called_once_with(
        start_epoch=expected_resumed_epoch,
        start_step=expected_resumed_global_step # Pass the loaded global step
    )
    # Check the result returned by train() matches the mock TrainingLoop.run return
    # Compare relevant parts, as trainer.train adds loop_result
    assert training_result["global_step"] == run_return_dict["final_global_step"]
    assert training_result["epoch"] == run_return_dict["epoch"]


    # 3. Check Callbacks orchestration
    mock_callback_list_instance.on_train_begin.assert_called_once_with(global_step=expected_resumed_global_step)
    # on_train_end should be called with the metrics from TrainingLoop.run, prefixed and augmented
    mock_callback_list_instance.on_train_end.assert_called_once()
    call_args, call_kwargs = mock_callback_list_instance.on_train_end.call_args
    end_metrics = call_kwargs.get('metrics', {})
    assert end_metrics['final_step'] == final_global_step_overall
    assert end_metrics['final_epoch'] == total_epochs - 1
    assert 'loop_loss' in end_metrics # Check prefixed key

    # 4. Check Evaluator calls for remaining epochs (logic moved to TrainingLoop, not checked here)

    # 5. Check CheckpointManager save calls (logic moved to TrainingLoop/callbacks, not checked here)

    # 6. Verify final Trainer state
    assert trainer.epoch == total_epochs - 1 # Last completed epoch index
    assert trainer.global_step == final_global_step_overall # Final step from TrainingLoop


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