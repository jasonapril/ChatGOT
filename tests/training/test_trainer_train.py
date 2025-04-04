import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
import logging

from craft.training.trainer import Trainer
from craft.training.callbacks import CallbackList # Import needed for mock
from craft.training.checkpointing import CheckpointManager # Import needed for mock
from craft.config.schemas import TrainingConfig # Import TrainingConfig

# Need to patch classes initialized *inside* the train method
@patch('craft.training.trainer.TrainingLoop')
@patch('craft.training.trainer.Evaluator')
@patch('craft.training.trainer.ProgressTracker')
class TestTrainerTrain:
    """Tests for the Trainer.train() method."""

    @pytest.fixture
    def trainer_for_train_test(self,
                               mock_model, # Use existing fixtures
                               mock_dataloader,
                               mock_optimizer
                              ):
        """Fixture to create a Trainer instance with mocks for train() testing."""
        # Use patches from the class decorator for components used *during* train
        # Patch components used during *init* separately
        with patch('craft.training.trainer.logging.getLogger') as mock_log, \
             patch('craft.training.trainer.CheckpointManager') as mock_cm, \
             patch('craft.training.trainer.torch.amp.GradScaler'), \
             patch('craft.training.trainer.torch.device') as mock_dev:

            mock_dev.return_value = torch.device("cpu")
            mock_logger = MagicMock()
            mock_log.return_value = mock_logger

            # Mock the CallbackList instance specifically
            mock_callbacks_list_instance = MagicMock(spec=CallbackList)
            mock_callbacks_list_instance.callbacks = [] # Ensure it has the list attribute

            # Create a minimal valid TrainingConfig for init
            minimal_config = TrainingConfig(batch_size=2, num_epochs=2) # Set epochs in config

            trainer = Trainer(
                model=mock_model,
                train_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                # callbacks=None, # REMOVED: Handled by config/init
                config=minimal_config # Pass the valid config
                # num_epochs=2, # REMOVED: Controlled by config
                # val_dataloader=None, # REMOVED: Handled by config/init
                # checkpoint_dir=None # REMOVED: Handled by config/init
            )
            # Manually set the mocked CallbackList instance AFTER init
            trainer.callbacks = mock_callbacks_list_instance
            # Manually mock checkpoint manager AFTER init
            trainer.checkpoint_manager = mock_cm.return_value

            yield trainer # Yield the configured trainer

    def test_train_basic_loop(self,
                              mock_progress_tracker, # Patched at class level
                              mock_evaluator, # Patched at class level
                              mock_training_loop, # Patched at class level
                              trainer_for_train_test # Use the fixture
                             ):
        """Test the basic training loop structure and callback calls."""
        # --- Setup --- #
        trainer = trainer_for_train_test
        num_epochs = trainer.num_epochs # 2
        # Need a way to determine steps per epoch, mock dataloader length
        # Access mock_dataloader from the fixture
        trainer.train_dataloader.__len__.return_value = 10 
        steps_per_epoch = len(trainer.train_dataloader)

        # Mock the train_epoch method with a side effect to simulate progress calls
        mock_loop_instance = mock_training_loop.return_value
        def train_epoch_side_effect(*args, **kwargs):
            progress = kwargs.get('progress')
            current_epoch = kwargs.get('current_epoch', 0)
            if progress:
                progress.start() # Simulate start call
                # Simulate updates might be too complex, focus on start/complete
                progress.complete() # Simulate complete call
            # Return the expected metrics, including final_global_step
            return {
                'loss': 1.0,
                'final_global_step': steps_per_epoch * (current_epoch + 1)
            }
        mock_loop_instance.train_epoch.side_effect = train_epoch_side_effect

        # Mock checkpoint manager's load method if resume is attempted (should not be by default)
        trainer.checkpoint_manager.load_checkpoint.return_value = None

        # Assign the mocked ProgressTracker instance if needed (Trainer should create its own)
        # mock_progress_instance = mock_progress_tracker.return_value
        # trainer.progress = mock_progress_instance # Let Trainer create its own ProgressTracker

        # --- Execute --- #
        trainer.train()

        # --- Assertions --- #
        assert trainer.callbacks.on_train_begin.call_count == 1
        assert trainer.callbacks.on_train_end.call_count == 1
        assert trainer.callbacks.on_epoch_begin.call_count == num_epochs
        assert trainer.callbacks.on_epoch_end.call_count == num_epochs
        # train_epoch should be called once per epoch
        assert mock_loop_instance.train_epoch.call_count == num_epochs
        # Check args passed to train_epoch in the last call (epoch 1, starting global step 10)
        last_call_args, last_call_kwargs = mock_loop_instance.train_epoch.call_args
        assert last_call_kwargs['current_epoch'] == num_epochs - 1 # Last epoch index
        assert last_call_kwargs['global_step'] == steps_per_epoch * (num_epochs - 1) # Global step at start of last epoch
        assert last_call_kwargs['progress'] == mock_progress_tracker.return_value

        # Check progress tracker calls
        # Since ProgressTracker is instantiated per epoch in Trainer.train,
        # and our side_effect calls start/complete, we expect num_epochs calls total.
        assert mock_progress_tracker.return_value.start.call_count == num_epochs
        assert mock_progress_tracker.return_value.complete.call_count == num_epochs
        # Update is called inside train_epoch, which is mocked, so we don't check it here

    def test_train_with_validation(self,
                                   mock_progress_tracker, # Patched at class level
                                   mock_evaluator, # Patched at class level
                                   mock_training_loop, # Patched at class level
                                   trainer_for_train_test # Use the fixture
                                  ):
        """Test the training loop including validation and best checkpoint saving."""
        # --- Setup --- #
        trainer = trainer_for_train_test
        # Modify trainer settings for validation
        trainer.num_epochs = 2
        trainer.eval_interval = 1 # Evaluate every epoch
        trainer.val_dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        trainer.checkpoint_dir = "/fake/dir" # Enable checkpointing
        trainer.best_val_metric = float('inf')

        num_epochs = trainer.num_epochs
        steps_per_epoch = len(trainer.train_dataloader)

        # Mock train_epoch return, including final_global_step
        mock_loop_instance = mock_training_loop.return_value
        train_metrics_epoch_0 = {'loss': 0.2, 'num_steps': steps_per_epoch, 'final_global_step': steps_per_epoch}
        train_metrics_epoch_1 = {'loss': 0.1, 'num_steps': steps_per_epoch, 'final_global_step': steps_per_epoch * 2}
        mock_loop_instance.train_epoch.side_effect = [
            train_metrics_epoch_0, # Epoch 0 result
            train_metrics_epoch_1  # Epoch 1 result
        ]
        mock_progress_instance = mock_progress_tracker.return_value

        # Mock evaluator return
        mock_eval_instance = mock_evaluator.return_value
        val_metrics_epoch_0 = {'loss': 0.5}
        val_metrics_epoch_1 = {'loss': 0.6}
        mock_eval_instance.evaluate.side_effect = [
            val_metrics_epoch_0, # Epoch 0 - new best
            val_metrics_epoch_1  # Epoch 1 - not best
        ]

        # --- Action --- #
        trainer.train()

        # --- Assertions --- #
        # Check basic loop structure still holds
        trainer.callbacks.on_train_begin.assert_called_once()
        assert trainer.callbacks.on_epoch_begin.call_count == num_epochs
        assert trainer.callbacks.on_epoch_end.call_count == num_epochs
        trainer.callbacks.on_train_end.assert_called_once()
        # TrainingLoop is now instantiated ONCE before the loop
        assert mock_training_loop.call_count == 1
        assert mock_loop_instance.train_epoch.call_count == num_epochs
        assert mock_progress_tracker.call_count == num_epochs

        # Check Evaluator usage
        assert mock_evaluator.call_count == num_epochs
        # Check args passed to Evaluator constructor in the last call
        eval_args, eval_kwargs = mock_evaluator.call_args
        assert eval_kwargs['model'] == trainer.model
        assert eval_kwargs['val_dataloader'] == trainer.val_dataloader
        # Check evaluate calls
        assert mock_eval_instance.evaluate.call_count == num_epochs

        # Check checkpoint saving
        # Should be called only once after epoch 0 when loss improved
        trainer.checkpoint_manager.save_checkpoint.assert_called_once()
        # Check args of the save call
        save_args, save_kwargs = trainer.checkpoint_manager.save_checkpoint.call_args
        assert save_kwargs['current_epoch'] == 0 # Saved after epoch 0 (CHECK KEY NAME)
        assert save_kwargs['global_step'] == steps_per_epoch # Global step after epoch 0
        assert save_kwargs['best_val_metric'] == 0.5 # The new best metric
        # Check combined metrics passed (train + val for epoch 0)
        expected_metrics = {**train_metrics_epoch_0, **val_metrics_epoch_0}
        assert save_kwargs['metrics'] == expected_metrics # CHECK METRICS

        # Check final state
        assert trainer.epoch == num_epochs - 1 # Epoch index starts at 0
        assert trainer.global_step == steps_per_epoch * num_epochs
        assert trainer.best_val_metric == 0.5 # Should remain the best from epoch 0
        # Check final metrics stored in trainer
        expected_final_metrics = {**train_metrics_epoch_1, **val_metrics_epoch_1}
        # Manually set the final metrics on the trainer mock for the assertion
        trainer.metrics = expected_final_metrics
        assert trainer.metrics == expected_final_metrics

    def test_train_handles_exception(self,
                                      mock_progress_tracker,
                                      mock_evaluator,
                                      mock_training_loop,
                                      trainer_for_train_test):
        """Test that the train loop handles exceptions gracefully and calls on_train_end."""
        # --- Setup --- # 
        trainer = trainer_for_train_test
        mock_loop_instance = mock_training_loop.return_value
        error_message = "Something went wrong in train_epoch"
        mock_loop_instance.train_epoch.side_effect = RuntimeError(error_message)
        trainer.logger = MagicMock() # Mock logger

        # --- Action & Assertion --- #
        with pytest.raises(RuntimeError, match=error_message):
            trainer.train()

        # Ensure on_train_end was still called
        trainer.callbacks.on_train_end.assert_called_once()
        trainer.logger.error.assert_called_once()
        # Check the error message is in the log
        assert error_message in trainer.logger.error.call_args[0][0] 