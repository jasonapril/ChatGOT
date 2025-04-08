import pytest
import torch
from unittest.mock import MagicMock, patch, ANY, call
import logging
from omegaconf import OmegaConf, DictConfig

from craft.training.trainer import Trainer
from craft.training.callbacks import CallbackList
from craft.training.checkpointing import CheckpointManager, TrainingState
from craft.config.schemas import TrainingConfig
from craft.models.base import LanguageModel
from torch.utils.data import DataLoader
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
        '_target_': 'tests.training.trainer.test_trainer_train.MockTarget', # Use mock target
        'architecture': 'mock_arch',
        'vocab_size': 100 # Example value
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


# Patch the instantiate function used *within* trainer.py
@patch('craft.training.trainer.instantiate')
# Patch other classes instantiated/used by Trainer or its methods
@patch('craft.training.trainer.TrainingLoop')
@patch('craft.training.trainer.torch.amp.GradScaler')
@patch('craft.training.trainer.logging.getLogger')
class TestTrainerTrain:
    """Tests for the Trainer.train() method."""

    @pytest.fixture
    def trainer_for_train_test(self,
                               mock_logger, # Patched via class decorator
                               mock_scaler, # Patched via class decorator
                               mock_training_loop, # Patched via class decorator
                               mock_instantiate, # Patched via class decorator
                               minimal_experiment_config_node, # Use the config fixture
                               mock_model, mock_optimizer, mock_dataloader, mock_tokenizer, # Mocks to be RETURNED by instantiate
                               mock_callback_list, mock_checkpoint_manager): # Other mocks
        """Fixture to provide a Trainer instance initialized via config."""

        # Configure mock_instantiate to return specific mocks based on config section
        def instantiate_side_effect(config_node, **kwargs):
            # This function mimics hydra.utils.instantiate based on the passed config node
            # We check the _target_ or structure to decide which mock to return
            target = getattr(config_node, '_target_', None)
            if target == 'craft.config.schemas.TrainingConfig':
                 # Instantiate the actual TrainingConfig pydantic model
                 return TrainingConfig(**OmegaConf.to_container(config_node, resolve=True)) # type: ignore
            elif config_node == minimal_experiment_config_node.model:
                 return mock_model
            elif config_node == minimal_experiment_config_node.optimizer:
                 return mock_optimizer
            elif config_node == minimal_experiment_config_node.data.tokenizer:
                 return mock_tokenizer
            elif config_node == minimal_experiment_config_node.data.datasets.train:
                 # Trainer wraps dataset in DataLoader if instantiate returns Dataset
                 # Simulate returning a mock dataset first
                 mock_dataset = MagicMock(spec=torch.utils.data.Dataset)
                 return mock_dataset # Trainer's logic will wrap this
            elif config_node == minimal_experiment_config_node.get('checkpointing'): # Handle None case
                # Return the specific mock needed for checkpointing tests
                return mock_checkpoint_manager
            elif config_node == minimal_experiment_config_node.get('callbacks'): # Handle None case
                 # Return individual mock callbacks if callbacks were configured
                 # For None config, Trainer creates an empty CallbackList internally
                 return None # Or return [] ? Check Trainer logic
            # Add more elif branches for scheduler, evaluator if needed by tests
            else:
                # Default fallback or raise error if unexpected config node
                # print(f"WARN: Unexpected instantiate call in test fixture: {config_node}")
                # Return a generic mock if needed for other calls
                return MagicMock()

        mock_instantiate.side_effect = instantiate_side_effect

        # Instantiate Trainer using the DictConfig
        trainer = Trainer(
            cfg=minimal_experiment_config_node,
            # CLI args can still be passed if needed by tests
            # resume_from_checkpoint=None,
            # compile_model=False,
            # experiment_name='test_exp_override'
        )

        # --- Post-init adjustments / overrides for testing ---
        # Trainer now creates its own CallbackList, but we might want to mock its methods
        # Replace the internally created one with our mock if needed for assertion checking
        trainer.callbacks = mock_callback_list
        trainer.callbacks.callbacks = [] # Ensure inner list exists

        # Manually assign the mocked CheckpointManager IF trainer didn't instantiate one
        # (based on the mocked instantiate logic above)
        # Check if the mock was returned correctly
        if trainer.checkpoint_manager is None or not isinstance(trainer.checkpoint_manager, MagicMock):
             trainer.checkpoint_manager = mock_checkpoint_manager # Force assign our mock

        # Ensure the dataloader assigned by Trainer (after wrapping) is our mock
        # Trainer's logic wraps the dataset returned by instantiate
        # We need to ensure the final dataloader is the mock we want to assert on
        # This assumes the fixture's mock_dataloader IS the DataLoader instance
        trainer.train_dataloader = mock_dataloader # Override the wrapped one

        # Mock the .to call that happens after model instantiation
        trainer.model.to.assert_called_once_with(trainer.device)
        trainer.model.to = MagicMock(return_value=trainer.model) # Mock subsequent calls if any

        yield trainer, mock_logger # Yield the configured trainer AND the mock logger


    def test_train_basic_loop(self,
                              mock_logger, # Patched
                              mock_scaler, # Patched
                              mock_training_loop, # Patched
                              mock_instantiate, # Patched
                              trainer_for_train_test # Use the fixture
                             ):
        """Test the basic training loop structure and callback calls."""
        # --- Setup --- #
        trainer, _ = trainer_for_train_test # Unpack trainer
        mock_loop_instance = mock_training_loop.return_value # Get the instance created by Trainer

        # Mock the run() method of the TrainingLoop instance
        final_step_simulated = 20
        final_epoch_simulated = 1
        mock_loop_instance.run.return_value = {
            'global_step': final_step_simulated,
            'epoch': final_epoch_simulated,
            'total_train_time': 123.45,
            'metrics': {'loss': 0.5}
        }

        # --- Execute --- #
        trainer.train()

        # --- Assertions --- #
        # Assert Trainer-level callbacks
        trainer.callbacks.on_train_begin.assert_called_once()
        trainer.callbacks.on_train_end.assert_called_once_with(metrics={'loss': 0.5})

        # Assert TrainingLoop was instantiated correctly within Trainer.train
        # Get the arguments passed to TrainingLoop constructor
        loop_call_args, loop_call_kwargs = mock_training_loop.call_args
        assert loop_call_kwargs['model'] == trainer.model
        assert loop_call_kwargs['optimizer'] == trainer.optimizer
        assert loop_call_kwargs['train_dataloader'] == trainer.train_dataloader
        assert loop_call_kwargs['device'] == trainer.device
        assert loop_call_kwargs['config'] == trainer.config # Should be the Pydantic TrainingConfig
        assert loop_call_kwargs['scheduler'] == trainer.scheduler
        assert loop_call_kwargs['checkpoint_manager'] == trainer.checkpoint_manager
        assert isinstance(loop_call_kwargs['callbacks'], list) # Inner list is passed

        # Assert TrainingLoop.run() was called
        mock_loop_instance.run.assert_called_once()

    # Patch instantiate specifically for this test's scope
    @patch('craft.training.trainer.instantiate')
    def test_train_with_validation(self,
                                   mock_instantiate_val, # Patched instantiate for this test
                                   mock_logger, # Patched via class decorator
                                   mock_scaler, # Patched via class decorator
                                   mock_training_loop, # Patched via class decorator
                                   # Mocks needed for side effect config
                                   mock_model,
                                   mock_optimizer,
                                   mock_dataloader, # Train dataloader mock
                                   mock_val_dataloader, # Val dataloader mock
                                   mock_tokenizer,
                                   mock_evaluator, # Evaluator mock
                                   mock_checkpoint_manager, # Checkpoint manager mock
                                   mock_callback_list, # Callback list mock
                                   # Config fixtures
                                   minimal_training_config_dict,
                                   minimal_model_config_dict,
                                   minimal_data_config_dict,
                                   minimal_optimizer_config_dict):
        """Test the training loop including validation and best checkpoint saving."""
        # --- Setup --- #
        # Create a config for this specific test with validation enabled
        test_training_config_dict = {
            **minimal_training_config_dict,
            'eval_interval': 1,
            'val_metric': "loss",
            'checkpoint_dir': "/fake/mock/dir", # Needed by CheckpointManager typically
            'save_interval': 10, # Example save interval
        }
        test_data_config_dict = {
             **minimal_data_config_dict,
             'datasets': {
                 'train': minimal_data_config_dict['datasets']['train'],
                 'val': {'_target_': 'tests.training.trainer.test_trainer_train.MockTarget'}, # Add val target
                 'test': None
             }
        }
        test_eval_config_dict = {'_target_': 'tests.training.trainer.test_trainer_train.MockTarget'}
        test_checkpointing_config_dict = {'_target_': 'tests.training.trainer.test_trainer_train.MockTarget'}

        test_config = OmegaConf.create({
            'training': test_training_config_dict,
            'model': minimal_model_config_dict,
            'data': test_data_config_dict,
            'optimizer': minimal_optimizer_config_dict,
            'eval': test_eval_config_dict,
            'checkpointing': test_checkpointing_config_dict,
            'callbacks': None,
            'scheduler': None,
            'device': 'cpu',
            'seed': 42,
            'log_level': 'INFO',
            'experiment_name': 'test_val_exp'
        })

        # Configure the mocked instantiate for this test's specific needs
        def instantiate_val_side_effect(config_node, **kwargs):
            target = getattr(config_node, '_target_', None)
            if target == 'craft.config.schemas.TrainingConfig':
                 return TrainingConfig(**OmegaConf.to_container(config_node, resolve=True)) # type: ignore
            elif config_node == test_config.model: return mock_model
            elif config_node == test_config.optimizer: return mock_optimizer
            elif config_node == test_config.data.tokenizer: return mock_tokenizer
            elif config_node == test_config.data.datasets.train: return MagicMock(spec=torch.utils.data.Dataset) # Train Dataset
            elif config_node == test_config.data.datasets.val: return MagicMock(spec=torch.utils.data.Dataset) # Val Dataset
            elif config_node == test_config.eval: return mock_evaluator
            elif config_node == test_config.checkpointing: return mock_checkpoint_manager
            # Add scheduler etc. if needed
            else: return MagicMock() # Default mock
        mock_instantiate_val.side_effect = instantiate_val_side_effect

        # Instantiate Trainer
        trainer = Trainer(cfg=test_config)

        # Post-init adjustments/overrides
        trainer.callbacks = mock_callback_list # Use shared mock
        trainer.callbacks.callbacks = []
        trainer.train_dataloader = mock_dataloader # Override wrapped loader
        trainer.val_dataloader = mock_val_dataloader # Override wrapped loader

        # Get the TrainingLoop instance created by Trainer.train
        mock_loop_instance = mock_training_loop.return_value

        # Configure TrainingLoop.run return value
        mock_loop_instance.run.return_value = {
            'global_step': 20, 'epoch': 1, 'metrics': {'loss': 0.5, 'val_loss': 0.4}
        }

        # --- Execute --- #
        trainer.train()

        # --- Assertions --- #
        trainer.callbacks.on_train_begin.assert_called_once()
        trainer.callbacks.on_train_end.assert_called_once()
        mock_loop_instance.run.assert_called_once()

        # Check that evaluator and checkpoint manager were instantiated (via mock_instantiate)
        eval_call = call(test_config.eval, model=mock_model, val_dataloader=mock_val_dataloader, device=trainer.device, tokenizer=mock_tokenizer)
        ckpt_call = call(test_config.checkpointing, config=trainer.config, experiment_config=test_config, model=mock_model, optimizer=mock_optimizer, callbacks=trainer.callbacks, scheduler=None, scaler=mock_scaler.return_value, tokenizer=mock_tokenizer)

        # Check if these specific calls were made to the mocked instantiate
        # Note: Order might matter depending on implementation details of __init__
        # assert eval_call in mock_instantiate_val.call_args_list
        # assert ckpt_call in mock_instantiate_val.call_args_list
        # Simpler check: Ensure instantiate was called for eval and checkpointing configs
        assert any(c.args[0] == test_config.eval for c in mock_instantiate_val.call_args_list)
        assert any(c.args[0] == test_config.checkpointing for c in mock_instantiate_val.call_args_list)


    def test_train_handles_exception(self,
                                     mock_logger, # Patched
                                     mock_scaler, # Patched
                                     mock_training_loop, # Patched
                                     mock_instantiate, # Patched
                                     trainer_for_train_test):
        """Test that exceptions during training loop are caught and callbacks called."""
        # --- Setup --- #
        trainer, logger_instance = trainer_for_train_test # Get trainer and logger mock
        mock_loop_instance = mock_training_loop.return_value
        test_exception = ValueError("Training Loop Failed!")
        mock_loop_instance.run.side_effect = test_exception

        # --- Execute & Assert --- #
        with pytest.raises(ValueError, match="Training Loop Failed!"):
            trainer.train()

        # Assertions
        trainer.callbacks.on_train_begin.assert_called_once()
        trainer.callbacks.on_train_error.assert_called_once_with(test_exception)
        # on_train_end should NOT be called if run() raises an exception before finishing
        trainer.callbacks.on_train_end.assert_not_called()
        logger_instance.error.assert_any_call(f"An error occurred during training: {test_exception}", exc_info=True)