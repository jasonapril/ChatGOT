import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from omegaconf import OmegaConf, DictConfig
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

# Define simple mock classes for target resolution if needed
class MockTarget:
    pass

# Minimal Config Fixtures
@pytest.fixture
def minimal_training_config_dict():
    # Return dict for easy merging into OmegaConf
    return {
        '_target_': 'craft.config.schemas.TrainingConfig', # Ensure it can be instantiated
        'batch_size': 2,
        'num_epochs': 2,
        'use_amp': False,
        'gradient_accumulation_steps': 1,
        'log_interval': 10,
        'save_interval': 0,
        'time_save_interval_seconds': 0,
        'resume_from_checkpoint': None,
        'max_steps': None,
        'max_grad_norm': None,
    }

@pytest.fixture
def minimal_model_config_dict():
    return {
        '_target_': 'unittest.mock.MagicMock',
        'config': { # Nested block
            'architecture': 'mock'
        }
    }

@pytest.fixture
def minimal_data_config_dict():
    # Structure matching how Trainer instantiates dataloaders
    return {
        'tokenizer': { '_target_': 'tests.training.trainer.test_trainer_train.MockTarget' },
        'datasets': {
            'train': { '_target_': 'tests.training.trainer.test_trainer_train.MockTarget' },
            'val': None,
            'test': None
        },
        'batch_size': 2, # Keep consistent with training config
        'num_workers': 0,
        'block_size': 16 # Example value
    }

@pytest.fixture
def minimal_optimizer_config_dict():
    return {'_target_': 'tests.training.trainer.test_trainer_train.MockTarget'}

@pytest.fixture
def minimal_experiment_config_node(minimal_training_config_dict,
                                     minimal_model_config_dict,
                                     minimal_data_config_dict,
                                     minimal_optimizer_config_dict):
    # Build the full DictConfig required by Trainer
    conf_dict = {
        'training': minimal_training_config_dict,
        'model': minimal_model_config_dict,
        'data': minimal_data_config_dict,
        'optimizer': minimal_optimizer_config_dict,
        'callbacks': None, # Trainer handles None
        'scheduler': None, # Trainer handles None
        'eval': None,      # Trainer handles None
        'checkpointing': None, # Trainer handles None
        'device': 'cpu',  # Specify device
        'seed': 42,       # Specify seed
        'log_level': 'INFO', # Specify log level
        'experiment_name': 'test_exp'
    }
    return OmegaConf.create(conf_dict)

@pytest.fixture
def setup_trainer_test_environment(minimal_model_config_dict):
    # --- Create TrainingConfig object --- #
    training_args = {
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
        "torch_compile": False,
        "sample_max_new_tokens": 100,
        "sample_temperature": 0.8,
        "sample_start_text": "Once upon a time",
        "val_metric": "loss",
        "time_save_interval_seconds": 0,
        "time_eval_interval_seconds": 0,
        "mixed_precision": False,
        "save_steps_interval": 0,
        "compile_model": False,
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
    ]
    mock_callback_list.callbacks = mock_callbacks_instance

    # --- Create Experiment Config Node (Mimicking Hydra Structure) --- #
    experiment_inner_dict = {
        'name': 'test_experiment',
        'output_dir': 'outputs/test_experiment',
        'device': training_args['device'],
        'training': OmegaConf.create(training_args),
        'model': OmegaConf.create(minimal_model_config_dict),
        'data': OmegaConf.create({
            'tokenizer': { '_target_': 'unittest.mock.MagicMock' },
            'batch_size': training_args['batch_size'],
            'num_workers': 0,
             'datasets': {
                'train': {
                    'dataset': { '_target_': 'unittest.mock.MagicMock' }
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
            'enable': False,
            'val_interval': 1
        }),
        'callbacks': {
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
        "mock_model": mock_model,
        "mock_optimizer": mock_optimizer,
        "mock_train_loader": mock_train_loader,
        "mock_val_loader": mock_val_loader,
        "mock_tokenizer": mock_tokenizer,
        "mock_scaler": mock_scaler,
        "mock_scheduler": mock_scheduler,
        "mock_checkpoint_manager": mock_checkpoint_manager,
        "mock_evaluator": mock_evaluator,
        "mock_tensorboard_logger": mock_tensorboard_logger,
        "mock_sample_generator": mock_sample_generator,
        "mock_callback_list": mock_callback_list,
    }

# Remove Old Fixtures
# @pytest.fixture
# def minimal_training_config_dict():
# ... (rest of old fixture)

# @pytest.fixture
# def minimal_model_config_dict():
# ... (rest of old fixture)

# @pytest.fixture
# def minimal_data_config_dict():
# ... (rest of old fixture)

# @pytest.fixture
# def minimal_optimizer_config_dict():
# ... (rest of old fixture)

# @pytest.fixture
# def minimal_experiment_config_node(...):
# ... (rest of old fixture)


# Remove Class Patches
# @patch('craft.training.trainer.instantiate')
# @patch('craft.training.trainer.TrainingLoop')
# @patch('craft.training.trainer.torch.amp.GradScaler')
# @patch('craft.training.trainer.logging.getLogger')
class TestTrainerTrain:
    """Tests for the Trainer.train() method."""

    # Remove Old Fixture
    # @pytest.fixture
    # def trainer_for_train_test(self, ...):
    # ... (rest of old fixture)

    # --- Tests will be refactored below --- #
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
    @patch("craft.training.trainer.CallbackList")
    def test_train_basic_loop(self,
                              # Patched initializers (in reverse order)
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
                              MockTrainingLoop, # Patched TrainingLoop class
                              # Fixture
                              setup_trainer_test_environment):
        """Test the basic training loop structure and callback calls (refactored)."""
        # --- Setup from Fixture --- #
        fixture_data = setup_trainer_test_environment
        experiment_config = fixture_data["experiment_config"]
        expected_training_config = fixture_data["expected_training_config"]

        # --- Configure mocks for patched initializers (similar to test_training.py) --- #
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
        mock_checkpoint_manager_instance = fixture_data["mock_checkpoint_manager"]
        mock_initialize_checkpoint_manager.return_value = mock_checkpoint_manager_instance
        mock_evaluator_instance = fixture_data["mock_evaluator"]
        mock_initialize_evaluator.return_value = mock_evaluator_instance
        mock_compile_model.side_effect = lambda model, *args, **kwargs: model

        # --- Configure mocks for Patched Classes --- #
        mock_training_loop_instance = MockTrainingLoop.return_value
        mock_callback_list_instance = MagicMock(spec=CallbackList)
        mock_callback_list_instance.callbacks = mock_raw_callback_list
        # Mock all required callback methods
        mock_callback_list_instance.on_train_begin = MagicMock()
        mock_callback_list_instance.on_epoch_begin = MagicMock()
        mock_callback_list_instance.on_step_begin = MagicMock() # Assuming TrainingLoop calls this
        mock_callback_list_instance.on_step_end = MagicMock()   # Assuming TrainingLoop calls this
        mock_callback_list_instance.on_validation_begin = MagicMock()
        mock_callback_list_instance.on_validation_end = MagicMock()
        mock_callback_list_instance.on_epoch_end = MagicMock()
        mock_callback_list_instance.on_train_end = MagicMock()
        mock_callback_list_instance.on_exception = MagicMock()
        MockCallbackList.return_value = mock_callback_list_instance

        # --- Configure TrainingLoop.train_epoch mock return --- #
        epoch_return_dict = {"loss": 0.5}
        mock_training_loop_instance.train_epoch.return_value = epoch_return_dict

        # --- Instantiate Trainer --- #
        trainer = Trainer(cfg=experiment_config)

        # --- Call train --- #
        training_result = trainer.train()

        # --- Assertions --- #
        # Assert TrainingLoop was instantiated correctly
        MockTrainingLoop.assert_called_once_with(
            model=fixture_data["mock_model"],
            optimizer=fixture_data["mock_optimizer"],
            train_dataloader=fixture_data["mock_train_loader"],
            device=mock_initialize_device.return_value,
            config=trainer.config,
            scheduler=fixture_data["mock_scheduler"],
            callbacks=mock_callback_list_instance,
            checkpoint_manager=mock_checkpoint_manager_instance
        )
        # Assert TrainingLoop.train_epoch() was called correct number of times
        assert mock_training_loop_instance.train_epoch.call_count == expected_training_config.num_epochs

        # Assert Trainer callbacks were called
        mock_callback_list_instance.on_train_begin.assert_called_once()
        assert mock_callback_list_instance.on_epoch_begin.call_count == expected_training_config.num_epochs
        # Add more specific callback call checks if needed
        mock_callback_list_instance.on_train_end.assert_called_once()

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
    @patch("craft.training.trainer.CallbackList")
    def test_train_with_validation(self,
                                   # Patched initializers (in reverse order)
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
                                   MockTrainingLoop, # Patched TrainingLoop class
                                   # Fixture
                                   setup_trainer_test_environment):
        """Test the training loop including validation setup (refactored)."""
        # --- Setup from Fixture --- #
        fixture_data = setup_trainer_test_environment
        experiment_config_copy = copy.deepcopy(fixture_data["experiment_config"])

        # --- Modify Config for Validation --- #
        experiment_config_copy.experiment.validation.enable = True
        experiment_config_copy.experiment.validation.val_interval = 1 # Eval every epoch
        experiment_config_copy.experiment.training.eval_interval = 1 # Ensure eval is triggered
        experiment_config_copy.experiment.training.save_interval = 0 # Disable saving for this test

        # --- Configure mocks for patched initializers --- #
        mock_initialize_device.return_value = torch.device("cpu")
        mock_initialize_tokenizer.return_value = fixture_data["mock_tokenizer"]
        mock_initialize_model.return_value = fixture_data["mock_model"]
        mock_initialize_dataloaders.return_value = (
            fixture_data["mock_train_loader"],
            fixture_data["mock_val_loader"] # Make sure val_loader is provided
        )
        mock_initialize_optimizer.return_value = fixture_data["mock_optimizer"]
        mock_initialize_scheduler.return_value = fixture_data["mock_scheduler"]
        mock_initialize_amp_scaler.return_value = fixture_data["mock_scaler"]
        mock_raw_callback_list = [
            fixture_data["mock_tensorboard_logger"],
            fixture_data["mock_sample_generator"]
        ]
        mock_initialize_callbacks.return_value = mock_raw_callback_list
        mock_checkpoint_manager_instance = fixture_data["mock_checkpoint_manager"]
        mock_initialize_checkpoint_manager.return_value = mock_checkpoint_manager_instance
        # Evaluator should be returned since validation is enabled
        mock_evaluator_instance = fixture_data["mock_evaluator"]
        mock_initialize_evaluator.return_value = mock_evaluator_instance
        mock_compile_model.side_effect = lambda model, *args, **kwargs: model

        # --- Configure mocks for Patched Classes --- #
        mock_training_loop_instance = MockTrainingLoop.return_value
        mock_callback_list_instance = MagicMock(spec=CallbackList)
        mock_callback_list_instance.callbacks = mock_raw_callback_list
        # Mock all required callback methods
        mock_callback_list_instance.on_train_begin = MagicMock()
        mock_callback_list_instance.on_epoch_begin = MagicMock()
        mock_callback_list_instance.on_step_begin = MagicMock()
        mock_callback_list_instance.on_step_end = MagicMock()
        mock_callback_list_instance.on_validation_begin = MagicMock()
        mock_callback_list_instance.on_validation_end = MagicMock()
        mock_callback_list_instance.on_epoch_end = MagicMock()
        mock_callback_list_instance.on_train_end = MagicMock()
        mock_callback_list_instance.on_exception = MagicMock()
        MockCallbackList.return_value = mock_callback_list_instance

        # --- Configure TrainingLoop.train_epoch mock return (including val metrics) --- #
        epoch_return_dict = {"loss": 0.5}
        mock_training_loop_instance.train_epoch.return_value = epoch_return_dict

        # --- Configure Evaluator mock --- #
        if mock_evaluator_instance:
            mock_evaluator_instance.evaluate.return_value = {"val_loss": 0.6}

        # --- Instantiate Trainer --- #
        trainer = Trainer(cfg=experiment_config_copy)

        # --- Call train --- #
        training_result = trainer.train()

        # --- Assertions --- #
        # Assert initialization happened correctly (ensure Evaluator was created)
        mock_initialize_evaluator.assert_called_once()
        assert trainer.evaluator is mock_evaluator_instance

        # Assert TrainingLoop was instantiated correctly
        MockTrainingLoop.assert_called_once_with(
            model=fixture_data["mock_model"],
            optimizer=fixture_data["mock_optimizer"],
            train_dataloader=fixture_data["mock_train_loader"],
            device=mock_initialize_device.return_value,
            config=trainer.config,
            scheduler=fixture_data["mock_scheduler"],
            callbacks=mock_callback_list_instance,
            checkpoint_manager=mock_checkpoint_manager_instance
        )
        # Assert TrainingLoop.train_epoch() was called for each epoch
        assert mock_training_loop_instance.train_epoch.call_count == experiment_config_copy.experiment.training.num_epochs

        # Assert Evaluator.evaluate was called (since eval_interval=1)
        if mock_evaluator_instance:
            assert mock_evaluator_instance.evaluate.call_count == experiment_config_copy.experiment.training.num_epochs

        # Assert callbacks related to validation were called
        assert mock_callback_list_instance.on_validation_begin.call_count == experiment_config_copy.experiment.training.num_epochs
        assert mock_callback_list_instance.on_validation_end.call_count == experiment_config_copy.experiment.training.num_epochs

        # Assert other callbacks
        mock_callback_list_instance.on_train_begin.assert_called_once()
        assert mock_callback_list_instance.on_epoch_begin.call_count == experiment_config_copy.experiment.training.num_epochs
        mock_callback_list_instance.on_train_end.assert_called_once()

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
    @patch("craft.training.trainer.CallbackList")
    def test_train_handles_exception(self,
                                     # Patched initializers (in reverse order)
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
                                     MockTrainingLoop, # Patched TrainingLoop class
                                     # Fixture
                                     setup_trainer_test_environment):
        """Test that exceptions during training loop are caught and callbacks called (refactored)."""
        # --- Setup from Fixture --- #
        fixture_data = setup_trainer_test_environment
        experiment_config = fixture_data["experiment_config"]

        # --- Configure mocks for patched initializers --- #
        # (Most return values don't matter for this specific test, but set them up)
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
        mock_raw_callback_list = [MagicMock(spec=Callback)]
        mock_initialize_callbacks.return_value = mock_raw_callback_list
        mock_checkpoint_manager_instance = fixture_data["mock_checkpoint_manager"]
        mock_initialize_checkpoint_manager.return_value = mock_checkpoint_manager_instance
        mock_evaluator_instance = fixture_data["mock_evaluator"]
        mock_initialize_evaluator.return_value = mock_evaluator_instance
        mock_compile_model.side_effect = lambda model, *args, **kwargs: model

        # --- Configure mocks for Patched Classes --- #
        mock_training_loop_instance = MockTrainingLoop.return_value
        mock_callback_list_instance = MagicMock(spec=CallbackList)
        # Add methods expected by Trainer.train()
        mock_callback_list_instance.on_train_begin = MagicMock()
        mock_callback_list_instance.on_epoch_begin = MagicMock()
        mock_callback_list_instance.on_step_begin = MagicMock()
        mock_callback_list_instance.on_step_end = MagicMock()
        mock_callback_list_instance.on_evaluation_begin = MagicMock()
        mock_callback_list_instance.on_evaluation_end = MagicMock()
        mock_callback_list_instance.on_epoch_end = MagicMock()
        mock_callback_list_instance.on_train_end = MagicMock()
        mock_callback_list_instance.on_exception = MagicMock()
        mock_callback_list_instance.callbacks = mock_raw_callback_list
        MockCallbackList.return_value = mock_callback_list_instance

        # --- Configure TrainingLoop.train_epoch to raise exception --- #
        test_exception = ValueError("Training Epoch Failed!")
        mock_training_loop_instance.train_epoch.side_effect = test_exception

        # --- Instantiate Trainer --- #
        trainer = Trainer(cfg=experiment_config)
        assert trainer.callbacks is mock_callback_list_instance # Verify mock setup

        # --- Execute & Assert --- #
        try:
            trainer.train()
        except Exception as e:
            pytest.fail(f"Trainer.train raised an unexpected exception: {e}")

        # Assert that on_exception was called on the callback list instance
        mock_callback_list_instance.on_exception.assert_called_once_with(exception=test_exception)
        # Assert train_end was still called
        mock_callback_list_instance.on_train_end.assert_called_once()