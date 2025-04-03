import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
import logging

# Import the class to test
from craft.training.trainer import Trainer
from craft.training.callbacks import CallbackList
from craft.training.checkpointing import CheckpointManager
from torch.cuda.amp import GradScaler as CudaGradScaler # Alias to avoid conflict if torch.amp is used

# --- Fixtures ---
# (We will add more specific fixtures as needed)

@pytest.fixture
def mock_model():
    model = MagicMock(spec=torch.nn.Module)
    model.to.return_value = model # Mock the .to() method
    return model

@pytest.fixture
def mock_train_dataloader():
    return MagicMock(spec=torch.utils.data.DataLoader)

@pytest.fixture
def mock_optimizer():
    return MagicMock(spec=torch.optim.Optimizer)

@pytest.fixture
def mock_device():
    # Simulate CPU device by default for simpler testing
    return torch.device("cpu")

# --- Test Class ---

class TestTrainerInit:
    """Tests for Trainer initialization."""

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    def test_init_defaults(self,
                           mock_torch_device,
                           mock_grad_scaler,
                           mock_checkpoint_manager,
                           mock_get_logger,
                           mock_model,
                           mock_train_dataloader,
                           mock_optimizer, # Need optimizer for CheckpointManager
                          ):
        """Test Trainer initialization with minimal required arguments."""
        # --- Setup Mocks ---
        mock_cpu_device = torch.device("cpu")
        mock_torch_device.return_value = mock_cpu_device
        mock_scaler_instance = MagicMock(spec=CudaGradScaler)
        mock_grad_scaler.return_value = mock_scaler_instance
        mock_logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger_instance
        mock_checkpoint_manager_instance = MagicMock(spec=CheckpointManager)
        mock_checkpoint_manager.return_value = mock_checkpoint_manager_instance

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_train_dataloader,
            optimizer=mock_optimizer, # Pass optimizer
            config={}
        )

        # --- Assertions ---
        assert trainer.model == mock_model
        assert trainer.train_dataloader == mock_train_dataloader
        assert trainer.val_dataloader is None
        assert trainer.optimizer == mock_optimizer
        assert trainer.scheduler is None
        assert trainer.config == {}
        assert trainer.device == mock_cpu_device
        assert trainer.checkpoint_dir is None
        assert trainer.use_amp is True # Default
        assert trainer.gradient_accumulation_steps == 1
        assert trainer.max_grad_norm == 1.0
        assert trainer.log_interval == 10
        assert trainer.eval_interval == 1000
        assert trainer.save_interval == 5000
        assert trainer.num_epochs == 1
        assert trainer.resume_from_checkpoint is None

        assert trainer.logger == mock_logger_instance
        mock_get_logger.assert_any_call('Trainer')
        # mock_get_logger.assert_any_call('CallbackList') # CallbackList likely gets its own logger

        assert trainer.callbacks is not None # Check callbacks list is initialized
        assert trainer.callbacks.callbacks == []

        mock_checkpoint_manager.assert_called_once_with(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=None,
            scaler=trainer.scaler,
            config={},
            checkpoint_dir=None,
            callbacks=trainer.callbacks,
            device=trainer.device
        )
        assert trainer.checkpoint_manager == mock_checkpoint_manager_instance

        assert trainer.epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_metric == float('inf')
        assert trainer.metrics == {}

        mock_model.to.assert_called_once_with(mock_cpu_device)

        # Check scaler initialization (using the deprecated path)
        mock_grad_scaler.assert_called_once_with(enabled=True)

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    @patch.object(Trainer, '_resume_from_checkpoint') # Patch the resume method
    def test_init_all_args(self,
                           mock_resume_method,
                           mock_torch_device,
                           mock_grad_scaler,
                           mock_checkpoint_manager,
                           mock_get_logger,
                           mock_model,
                           mock_train_dataloader,
                           mock_optimizer,
                          ):
        """Test Trainer initialization with most arguments provided."""
        # --- Setup Mocks & Args ---
        mock_gpu_device = torch.device("cuda")
        mock_torch_device.return_value = mock_gpu_device

        # Configure the mock GradScaler instance BEFORE Trainer init uses it
        mock_scaler_instance = mock_grad_scaler.return_value
        # When use_amp=False, is_enabled() should return False
        mock_scaler_instance.is_enabled.return_value = False

        mock_logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger_instance
        mock_checkpoint_manager_instance = MagicMock(spec=CheckpointManager)
        mock_checkpoint_manager.return_value = mock_checkpoint_manager_instance

        mock_val_dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
        mock_callback = MagicMock()
        callbacks = [mock_callback]
        config = {'key': 'value', 'lr': 0.01}
        checkpoint_dir = "/tmp/checkpoints"
        resume_path = "/tmp/checkpoints/latest"
        use_amp = False
        grad_accum = 4
        max_grad = 0.5
        log_int = 50
        eval_int = 500
        save_int = 1000
        epochs = 5

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_train_dataloader,
            val_dataloader=mock_val_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=config,
            device=mock_gpu_device, # Explicitly pass device
            callbacks=callbacks,
            checkpoint_dir=checkpoint_dir,
            use_amp=use_amp,
            gradient_accumulation_steps=grad_accum,
            max_grad_norm=max_grad,
            log_interval=log_int,
            eval_interval=eval_int,
            save_interval=save_int,
            num_epochs=epochs,
            resume_from_checkpoint=resume_path
        )

        # --- Assertions ---
        assert trainer.model == mock_model
        assert trainer.train_dataloader == mock_train_dataloader
        assert trainer.val_dataloader == mock_val_dataloader
        assert trainer.optimizer == mock_optimizer
        assert trainer.scheduler == mock_scheduler
        assert trainer.config == config
        assert trainer.device == mock_gpu_device
        assert trainer.checkpoint_dir == checkpoint_dir
        assert trainer.use_amp == use_amp
        assert trainer.gradient_accumulation_steps == grad_accum
        assert trainer.max_grad_norm == max_grad
        assert trainer.log_interval == log_int
        assert trainer.eval_interval == eval_int
        assert trainer.save_interval == save_int
        assert trainer.num_epochs == epochs
        assert trainer.resume_from_checkpoint == resume_path

        assert trainer.logger == mock_logger_instance
        mock_get_logger.assert_any_call('Trainer')
        # mock_get_logger.assert_any_call('CallbackList') # CallbackList likely gets its own logger

        assert isinstance(trainer.callbacks, CallbackList)
        assert trainer.callbacks.callbacks == callbacks

        mock_checkpoint_manager.assert_called_once_with(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            scaler=trainer.scaler,
            config=config,
            checkpoint_dir=checkpoint_dir,
            callbacks=trainer.callbacks,
            device=trainer.device
        )
        assert trainer.checkpoint_manager == mock_checkpoint_manager_instance

        # Check scaler is initialized correctly based on use_amp=False
        mock_grad_scaler.assert_called_once_with(enabled=False) # Should be called but disabled
        assert trainer.scaler is mock_scaler_instance # Check the instance was used
        assert not trainer.scaler.is_enabled() # Explicit check using configured mock

        mock_model.to.assert_called_once_with(mock_gpu_device)

        # Check resume was called
        mock_resume_method.assert_called_once() 

class TestTrainerResume:
    """Tests for the Trainer._resume_from_checkpoint method."""

    @pytest.fixture
    def trainer_instance(self, mock_model, mock_train_dataloader, mock_optimizer):
        """Fixture to create a Trainer instance with minimal mocks for resume testing."""
        with patch('craft.training.trainer.logging.getLogger'), \
             patch('craft.training.trainer.CheckpointManager') as mock_cm, \
             patch('craft.training.trainer.torch.amp.GradScaler'), \
             patch('craft.training.trainer.torch.device') as mock_dev:

            mock_dev.return_value = torch.device("cpu")
            # Prevent actual resume call during initialization for these tests
            trainer = Trainer(
                model=mock_model,
                train_dataloader=mock_train_dataloader,
                optimizer=mock_optimizer,
                resume_from_checkpoint=None # Don't trigger resume in init
            )
            # Replace manager instance after init
            trainer.checkpoint_manager = mock_cm()
            return trainer

    def test_resume_successful(self, trainer_instance):
        """Test successful resumption from a checkpoint."""
        # --- Setup --- #
        resume_path = "/path/to/checkpoint"
        state = {
            'epoch': 5,
            'global_step': 1000,
            'best_val_metric': 0.5,
            'metrics': {'loss': 0.6}
        }
        trainer_instance.checkpoint_manager.load_checkpoint.return_value = state
        trainer_instance.resume_from_checkpoint = resume_path # Set path to trigger resume
        trainer_instance.logger = MagicMock() # Mock logger for assertion

        # --- Action --- #
        trainer_instance._resume_from_checkpoint()

        # --- Assertions --- #
        trainer_instance.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)
        assert trainer_instance.epoch == state['epoch']
        assert trainer_instance.global_step == state['global_step']
        assert trainer_instance.best_val_metric == state['best_val_metric']
        assert trainer_instance.metrics == state['metrics']
        trainer_instance.logger.info.assert_called_once_with(
            f"Resumed trainer state from checkpoint at epoch {state['epoch']}, step {state['global_step']}"
        )

    def test_resume_failure(self, trainer_instance):
        """Test failure during checkpoint loading."""
        # --- Setup --- #
        resume_path = "/path/to/bad/checkpoint"
        error_message = "File not found"
        trainer_instance.checkpoint_manager.load_checkpoint.side_effect = FileNotFoundError(error_message)
        trainer_instance.resume_from_checkpoint = resume_path
        trainer_instance.logger = MagicMock()

        # --- Action & Assertions --- #
        with pytest.raises(FileNotFoundError, match=error_message):
            trainer_instance._resume_from_checkpoint()

        trainer_instance.checkpoint_manager.load_checkpoint.assert_called_once_with(resume_path)
        trainer_instance.logger.error.assert_called_once()
        # Check that the error message logged contains the exception string
        assert error_message in trainer_instance.logger.error.call_args[0][0]


# --- Tests for train() method ---

# Need to patch classes initialized *inside* the train method
@patch('craft.training.trainer.TrainingLoop')
@patch('craft.training.trainer.Evaluator')
@patch('craft.training.trainer.ProgressTracker')
class TestTrainerTrain:
    """Tests for the Trainer.train() method."""

    @pytest.fixture
    def trainer_for_train_test(self,
                               mock_model, # Use existing fixtures
                               mock_train_dataloader,
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

            trainer = Trainer(
                model=mock_model,
                train_dataloader=mock_train_dataloader,
                optimizer=mock_optimizer,
                callbacks=None, # Pass None initially, not the mock instance
                num_epochs=2, # Run for 2 epochs for testing
                val_dataloader=None, # Disable validation for basic test
                checkpoint_dir=None # Disable checkpointing for basic test
            )
            trainer.logger = mock_logger # Assign mocked logger
            trainer.callbacks = mock_callbacks_list_instance # Assign mock AFTER init for assertions
            trainer.checkpoint_manager = mock_cm() # Assign mocked manager
            # Mock dataloader length
            mock_train_dataloader.__len__.return_value = 10
            return trainer

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
        steps_per_epoch = len(trainer.train_dataloader) # 10

        # Mock the return value of train_epoch, including final_global_step
        mock_loop_instance = mock_training_loop.return_value
        # Need a side effect to return correct final_global_step per epoch
        def train_epoch_side_effect(*args, **kwargs):
            start_step = kwargs.get('global_step', 0)
            return {'loss': 0.1, 'num_steps': steps_per_epoch, 'final_global_step': start_step + steps_per_epoch}
        mock_loop_instance.train_epoch.side_effect = train_epoch_side_effect

        mock_progress_instance = mock_progress_tracker.return_value

        # --- Action --- #
        trainer.train()

        # --- Assertions --- #
        # Check top-level callbacks
        trainer.callbacks.on_train_begin.assert_called_once()
        assert trainer.callbacks.on_epoch_begin.call_count == num_epochs
        assert trainer.callbacks.on_epoch_end.call_count == num_epochs
        trainer.callbacks.on_train_end.assert_called_once()

        # Check TrainingLoop usage
        assert mock_training_loop.call_count == num_epochs
        # Check args passed to TrainingLoop constructor in the last call
        loop_args, loop_kwargs = mock_training_loop.call_args
        assert loop_kwargs['model'] == trainer.model
        assert loop_kwargs['optimizer'] == trainer.optimizer
        assert loop_kwargs['train_dataloader'] == trainer.train_dataloader
        assert loop_kwargs['callbacks'] == trainer.callbacks # Check mock CallbackList passed

        # Check train_epoch calls
        assert mock_loop_instance.train_epoch.call_count == num_epochs
        # Check args passed to train_epoch in the last call (epoch 1, starting global step 10)
        epoch_args, epoch_kwargs = mock_loop_instance.train_epoch.call_args
        assert epoch_kwargs['current_epoch'] == num_epochs - 1 # Last epoch index
        assert epoch_kwargs['global_step'] == steps_per_epoch * (num_epochs - 1) # Global step at start of last epoch
        assert epoch_kwargs['progress'] == mock_progress_instance

        # Check ProgressTracker usage
        assert mock_progress_tracker.call_count == num_epochs
        # Check args passed to ProgressTracker constructor in the last call
        progress_args, progress_kwargs = mock_progress_tracker.call_args
        assert progress_kwargs['total_steps'] == steps_per_epoch
        assert progress_kwargs['desc'] == f"Epoch {num_epochs}/{num_epochs}" # Last epoch

        # Check final state
        assert trainer.epoch == num_epochs - 1 # Epoch index is 0-based
        assert trainer.global_step == steps_per_epoch * num_epochs

        # Ensure evaluation and saving were not attempted
        mock_evaluator.assert_not_called()
        trainer.checkpoint_manager.save_checkpoint.assert_not_called()

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
        assert mock_training_loop.call_count == num_epochs
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
        assert trainer.epoch == num_epochs - 1
        assert trainer.global_step == steps_per_epoch * num_epochs
        assert trainer.best_val_metric == 0.5 # Should retain the best metric found

    def test_train_with_save_interval(self,
                                      mock_progress_tracker, # Patched at class level
                                      mock_evaluator, # Patched at class level
                                      mock_training_loop, # Patched at class level
                                      trainer_for_train_test # Use the fixture
                                     ):
        """Test the training loop including interval-based checkpoint saving.
           NOTE: Trainer delegates step-based saving to TrainingLoop.
           This test verifies Trainer doesn't incorrectly save based on epoch interval.
        """
        # --- Setup --- #
        trainer = trainer_for_train_test
        # Modify trainer settings for saving
        trainer.num_epochs = 2
        trainer.save_interval = 1 # Set interval, but Trainer shouldn't use it directly
        trainer.val_dataloader = None # Disable validation
        trainer.checkpoint_dir = "/fake/dir/save" # Enable checkpointing
        trainer.eval_interval = 1000 # Ensure eval doesn't trigger save

        num_epochs = trainer.num_epochs
        steps_per_epoch = len(trainer.train_dataloader)

        # Mock train_epoch return, including final_global_step
        mock_loop_instance = mock_training_loop.return_value
        train_metrics_epoch_0 = {'loss': 0.2, 'num_steps': steps_per_epoch, 'final_global_step': steps_per_epoch}
        train_metrics_epoch_1 = {'loss': 0.1, 'num_steps': steps_per_epoch, 'final_global_step': steps_per_epoch * 2}
        mock_loop_instance.train_epoch.side_effect = [
            train_metrics_epoch_0,
            train_metrics_epoch_1
        ]
        mock_progress_instance = mock_progress_tracker.return_value

        # --- Action --- #
        trainer.train()

        # --- Assertions --- #
        # Check basic loop structure
        assert mock_loop_instance.train_epoch.call_count == num_epochs

        # Check Checkpoint saving
        # Trainer should NOT save based on its save_interval, it delegates to TrainingLoop
        trainer.checkpoint_manager.save_checkpoint.assert_not_called()
        mock_evaluator.assert_not_called() # Ensure validation didn't run

        # Check final state (ensure global step updated correctly)
        assert trainer.epoch == num_epochs - 1
        assert trainer.global_step == steps_per_epoch * num_epochs

    def test_train_handles_exception(self,
                                     mock_progress_tracker, # Patched at class level
                                     mock_evaluator, # Patched at class level
                                     mock_training_loop, # Patched at class level
                                     trainer_for_train_test # Use the fixture
                                    ):
        """Test that train() handles exceptions during loop and calls on_train_end."""
        # --- Setup --- #
        trainer = trainer_for_train_test
        trainer.num_epochs = 2
        trainer.val_dataloader = None
        trainer.checkpoint_dir = None

        mock_loop_instance = mock_training_loop.return_value
        error_message = "Training loop crashed!"
        mock_loop_instance.train_epoch.side_effect = RuntimeError(error_message)

        # --- Action & Assertions --- #
        with pytest.raises(RuntimeError, match=error_message):
            trainer.train()

        # Check that train_end was still called
        trainer.callbacks.on_train_end.assert_called_once()

        # Check that the error was logged
        trainer.logger.error.assert_called_once()
        assert error_message in trainer.logger.error.call_args[0][0]

        # Check that the loop stopped early (only first epoch attempted)
        mock_loop_instance.train_epoch.assert_called_once()
        trainer.callbacks.on_epoch_begin.assert_called_once()
        trainer.callbacks.on_epoch_end.assert_not_called() # Should not be called if epoch errors


# --- Test for generate_text --- #

@patch('craft.training.trainer.TextGenerator')
class TestTrainerGenerate:
    """Tests for the Trainer.generate_text method."""

    def test_generate_text_passes_args(self, mock_text_generator, mock_model, mock_train_dataloader, mock_optimizer):
        """Test that generate_text initializes TextGenerator and passes args."""
        # --- Setup --- #
        # Create a minimal trainer instance
        with patch('craft.training.trainer.logging.getLogger'), \
             patch('craft.training.trainer.CheckpointManager'), \
             patch('craft.training.trainer.torch.amp.GradScaler'), \
             patch('craft.training.trainer.torch.device') as mock_dev:
            mock_cpu_device = torch.device("cpu")
            mock_dev.return_value = mock_cpu_device
            trainer = Trainer(
                model=mock_model,
                train_dataloader=mock_train_dataloader,
                optimizer=mock_optimizer,
                config={'some_config': 123} # Include some config
            )

        mock_generator_instance = mock_text_generator.return_value
        mock_generator_instance.generate_text.return_value = ["generated text"]

        # --- Args for generate_text --- #
        prompt = "Hello"
        max_new = 50
        temp = 0.7
        top_k = 30
        top_p = 0.8
        rep_pen = 1.1
        num_ret = 2
        use_beam = True
        num_beams = 4
        len_pen = 1.2
        early_stop = False

        # --- Action --- #
        result = trainer.generate_text(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_pen,
            num_return_sequences=num_ret,
            use_beam_search=use_beam,
            num_beams=num_beams,
            length_penalty=len_pen,
            early_stopping=early_stop
        )

        # --- Assertions --- #
        # Check TextGenerator initialization
        mock_text_generator.assert_called_once_with(
            model=trainer.model,
            device=trainer.device,
            config=trainer.config
        )

        # Check generate_text call on the instance
        mock_generator_instance.generate_text.assert_called_once_with(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_pen,
            num_return_sequences=num_ret,
            use_beam_search=use_beam,
            num_beams=num_beams,
            length_penalty=len_pen,
            early_stopping=early_stop
        )

        # Check result
        assert result == ["generated text"] 