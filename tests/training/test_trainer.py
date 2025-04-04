import pytest
import torch
from unittest.mock import MagicMock, patch, ANY
import logging
from pydantic import ValidationError

# Import the class to test
from craft.training.trainer import Trainer
from craft.training.callbacks import CallbackList
from craft.training.checkpointing import CheckpointManager
from torch.cuda.amp import GradScaler as CudaGradScaler # Alias to avoid conflict if torch.amp is used
from craft.config.schemas import TrainingConfig, AppConfig # Import TrainingConfig and AppConfig
from craft.data.tokenizers.base import BaseTokenizer # Import BaseTokenizer

# --- Fixtures ---
# (We will add more specific fixtures as needed)

@pytest.fixture
def mock_model():
    m = MagicMock(spec=torch.nn.Module)
    m.to.return_value = m
    return m

@pytest.fixture
def mock_dataloader():
    return MagicMock(spec=torch.utils.data.DataLoader)

@pytest.fixture
def mock_optimizer():
    return MagicMock(spec=torch.optim.Optimizer)

@pytest.fixture
def mock_scheduler():
    return MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)

@pytest.fixture
def mock_tokenizer():
    return MagicMock(spec=BaseTokenizer)

@pytest.fixture
def mock_callback():
    return MagicMock() # Simple mock for callbacks

@pytest.fixture
def default_training_config() -> TrainingConfig:
    """Provides a default, valid TrainingConfig instance."""
    # Minimal valid config according to Pydantic model defaults
    # Add required fields that don't have defaults
    return TrainingConfig(batch_size=8) # Add a default batch_size

@pytest.fixture
def full_training_config() -> TrainingConfig:
    """Provides a TrainingConfig with more fields set."""
    return TrainingConfig(
        epochs=5,
        use_amp=False,
        gradient_accumulation_steps=4,
        max_grad_norm=0.5,
        log_interval=50,
        save_steps_interval=1000,
        time_save_interval_seconds=3600,
        eval_interval=500,
        compile_model=False,
        activation_checkpointing=False,
        torch_compile=False,
        batch_size=16, # Included for completeness
        learning_rate=1e-4, # Included for completeness
        max_steps=10000, # Included for completeness
    )

# --- Test Class ---

class TestTrainerInit:
    """Tests for Trainer initialization."""

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    def test_init_minimal_required(self,
                           mock_torch_device,
                           mock_grad_scaler,
                           mock_checkpoint_manager,
                           mock_get_logger,
                           mock_model,
                           mock_dataloader,
                           mock_optimizer,
                           default_training_config # Use fixture
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
        
        # Mock config model_dump
        mock_config_dict = default_training_config.model_dump()


        # --- Initialize Trainer ---
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=default_training_config # Pass TrainingConfig object
        )

        # --- Assertions ---
        assert trainer.model == mock_model
        assert trainer.train_dataloader == mock_dataloader
        assert trainer.val_dataloader is None
        assert trainer.optimizer == mock_optimizer
        assert trainer.scheduler is None
        assert trainer.config == default_training_config # Check config object stored
        assert trainer.device == mock_cpu_device
        # assert trainer.checkpoint_dir is None # REMOVED checkpoint_dir
        assert trainer.use_amp == default_training_config.use_amp # Check derived from config
        assert trainer.gradient_accumulation_steps == default_training_config.gradient_accumulation_steps
        assert trainer.max_grad_norm == default_training_config.max_grad_norm
        assert trainer.log_interval == default_training_config.log_interval
        assert trainer.eval_interval == default_training_config.eval_interval
        assert trainer.save_interval == default_training_config.save_interval # Should compare against save_interval
        assert trainer.num_epochs == default_training_config.num_epochs
        assert trainer.resume_from_checkpoint is None

        assert trainer.logger == mock_logger_instance
        mock_get_logger.assert_any_call('Trainer')

        assert trainer.callbacks is not None
        assert trainer.callbacks.callbacks == []

        # Check CheckpointManager call (no checkpoint_dir, uses config.model_dump())
        mock_checkpoint_manager.assert_called_once_with(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=None,
            scaler=trainer.scaler,
            config=mock_config_dict, # Expects dumped dict
            # checkpoint_dir=None, # REMOVED
            callbacks=trainer.callbacks,
            device=trainer.device,
            tokenizer=None # Default tokenizer is None
        )
        assert trainer.checkpoint_manager == mock_checkpoint_manager_instance

        assert trainer.epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_metric == float('inf')
        assert trainer.metrics == {}

        mock_model.to.assert_called_once_with(mock_cpu_device)

        # Check scaler initialization (using the correct path with device type)
        mock_grad_scaler.assert_called_once_with(enabled=default_training_config.use_amp)

        assert trainer.compile_model == default_training_config.compile_model # Check default

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    @patch.object(Trainer, '_resume_from_checkpoint') # Patch the resume method
    def test_init_all_args_provided(self,
                           mock_resume_method,
                           mock_torch_device,
                           mock_grad_scaler,
                           mock_checkpoint_manager,
                           mock_get_logger,
                           mock_model,
                           mock_dataloader,
                           mock_optimizer,
                           mock_scheduler,
                           mock_tokenizer,
                           mock_callback,
                           full_training_config # Use the more detailed config fixture
                          ):
        """Test Trainer initialization with most arguments provided."""
        # --- Setup Mocks & Args ---
        mock_gpu_device = torch.device("cuda")
        mock_torch_device.return_value = mock_gpu_device

        mock_scaler_instance = mock_grad_scaler.return_value
        # Test with use_amp = False
        mock_scaler_instance.is_enabled.return_value = False 
        use_amp_test_value = False 

        mock_logger_instance = MagicMock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger_instance
        mock_checkpoint_manager_instance = MagicMock(spec=CheckpointManager)
        mock_checkpoint_manager.return_value = mock_checkpoint_manager_instance
    
        mock_val_dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        mock_scheduler = MagicMock(spec=torch.optim.lr_scheduler._LRScheduler)
        mock_callback = MagicMock()
        callbacks = [mock_callback]
        mock_tokenizer = MagicMock() # Mock a tokenizer
        resume_path = "/tmp/checkpoints/latest"

        # Use the fixture directly instead of creating a new TrainingConfig instance here
        # test_config = TrainingConfig(
        #     use_amp=use_amp_test_value,
        #     gradient_accumulation_steps=4,
        #     max_grad_norm=0.5,
        #     log_interval=50,
        #     eval_interval=500,
        #     save_steps_interval=1000, # Use correct field name
        #     epochs=5,
        #     batch_size=16 # Ensure batch_size is present
        # )
        # Use the provided fixture which should already be valid
        test_config = full_training_config
        test_config.use_amp = use_amp_test_value # Override specific value if needed

        # --- Initialize Trainer ---
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            val_dataloader=mock_val_dataloader,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            config=test_config, # Pass the config object
            device=mock_gpu_device, # Explicitly pass device
            callbacks=callbacks,
            tokenizer=mock_tokenizer, # Pass tokenizer
            # checkpoint_dir=checkpoint_dir, # REMOVED
            # use_amp=use_amp, # REMOVED - now from config
            # gradient_accumulation_steps=grad_accum, # REMOVED
            # max_grad_norm=max_grad, # REMOVED
            # log_interval=log_int, # REMOVED
            # eval_interval=eval_int, # REMOVED
            # save_interval=save_int, # REMOVED
            # num_epochs=epochs, # REMOVED
            resume_from_checkpoint=resume_path
        )

        # --- Assertions ---
        assert trainer.model == mock_model
        assert trainer.train_dataloader == mock_dataloader
        assert trainer.val_dataloader == mock_val_dataloader
        assert trainer.optimizer == mock_optimizer
        assert trainer.scheduler == mock_scheduler
        assert trainer.config == test_config # Check config object stored
        assert trainer.device == mock_gpu_device
        assert trainer.tokenizer == mock_tokenizer
        # assert trainer.checkpoint_dir == checkpoint_dir # REMOVED
        # Check derived attributes against the config used
        assert trainer.use_amp == test_config.use_amp 
        assert trainer.gradient_accumulation_steps == test_config.gradient_accumulation_steps
        assert trainer.max_grad_norm == test_config.max_grad_norm
        assert trainer.log_interval == test_config.log_interval
        assert trainer.eval_interval == test_config.eval_interval
        assert trainer.save_interval == test_config.save_interval # Use the correct config field
        assert trainer.num_epochs == test_config.num_epochs
        assert trainer.resume_from_checkpoint == resume_path

        assert isinstance(trainer.callbacks, CallbackList)
        assert trainer.callbacks.callbacks == callbacks # Check if callbacks were wrapped

        # Check CheckpointManager call
        mock_config_dict = test_config.model_dump()
        mock_checkpoint_manager.assert_called_once_with(
            model=mock_model,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            scaler=trainer.scaler,
            config=mock_config_dict, # Pass dumped dict
            # checkpoint_dir=checkpoint_dir, # REMOVED
            callbacks=trainer.callbacks,
            device=trainer.device,
            tokenizer=mock_tokenizer # Check tokenizer passed
        )
        assert trainer.checkpoint_manager == mock_checkpoint_manager_instance

        # Check if resume was called because resume_from_checkpoint was provided
        mock_resume_method.assert_called_once()

        # Check scaler initialization
        # Check only 'enabled' as device type isn't passed directly in init
        mock_grad_scaler.assert_called_once_with(enabled=test_config.use_amp)

        assert trainer.compile_model == test_config.compile_model # Check compile model flag stored

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    @patch('craft.training.trainer.torch.cuda.is_available')
    def test_init_auto_device_detection(self, mock_cuda_available, mock_torch_device, mock_grad_scaler, mock_checkpoint_manager, mock_get_logger,
                                      mock_model, mock_dataloader, mock_optimizer, default_training_config):
        """Test automatic device detection (CPU and CUDA)."""
        # Test CPU case
        mock_cuda_available.return_value = False
        with mock_cuda_available:
            trainer_cpu = Trainer(model=mock_model, train_dataloader=mock_dataloader, optimizer=mock_optimizer, config=default_training_config)
        assert trainer_cpu.device == torch.device("cpu")
        # Check the arguments passed to the GradScaler class during init
        mock_grad_scaler.assert_called_with(enabled=default_training_config.use_amp)

        # Test CUDA case
        mock_cuda_available.return_value = True
        mock_torch_device.return_value = torch.device("cuda") # Ensure device returns cuda
        with mock_cuda_available:
            trainer_cuda = Trainer(model=mock_model, train_dataloader=mock_dataloader, optimizer=mock_optimizer, config=default_training_config)
        assert trainer_cuda.device == torch.device("cuda")
        # Check the arguments passed to the GradScaler class during init
        mock_grad_scaler.assert_called_with(enabled=default_training_config.use_amp)

class TestTrainerResume:
    """Tests for Trainer checkpoint resuming logic."""

    @pytest.fixture
    def trainer_instance(self, mock_model, mock_dataloader, mock_optimizer):
        """Fixture to create a Trainer instance with minimal mocks for resume testing."""
        with patch('craft.training.trainer.logging.getLogger'), \
             patch('craft.training.trainer.CheckpointManager') as mock_cm, \
             patch('craft.training.trainer.torch.amp.GradScaler'), \
             patch('craft.training.trainer.torch.device') as mock_dev:

            mock_dev.return_value = torch.device("cpu")
            # Prevent actual resume call during initialization for these tests
            # Create a minimal valid config
            minimal_config = TrainingConfig(batch_size=4)
            trainer = Trainer(
                model=mock_model,
                train_dataloader=mock_dataloader,
                optimizer=mock_optimizer,
                config=minimal_config # Pass the config object
                # resume_from_checkpoint=None # REMOVED: Handled by config
            )
            # Manually assign the mocked checkpoint manager AFTER init
            trainer.checkpoint_manager = mock_cm.return_value
            return trainer

    def test_resume_successful(self, trainer_instance, tmp_path):
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

    @patch('craft.training.trainer.logging.getLogger')
    @patch('craft.training.trainer.CheckpointManager')
    @patch('craft.training.trainer.torch.amp.GradScaler')
    @patch('craft.training.trainer.torch.device')
    def test_generate_text_passes_args(self, mock_torch_device, mock_grad_scaler, mock_checkpoint_manager, mock_get_logger, mock_text_generator, mock_model, mock_dataloader, mock_optimizer, default_training_config):
        """Test that generate_text initializes TextGenerator and passes args."""
        # --- Setup --- #
        # Mock device
        mock_cpu_device = torch.device("cpu")
        mock_torch_device.return_value = mock_cpu_device

        # Mock TrainingConfig and its model_dump
        mock_config_obj = MagicMock(spec=TrainingConfig)
        # Mock attributes accessed by Trainer.__init__
        mock_config_obj.num_epochs = 1
        mock_config_obj.max_steps = None
        mock_config_obj.use_amp = False
        mock_config_obj.gradient_accumulation_steps = 1
        mock_config_obj.max_grad_norm = None
        mock_config_obj.log_interval = 10
        mock_config_obj.eval_interval = 100
        mock_config_obj.save_interval = 500
        mock_config_obj.save_steps_interval = 0
        mock_config_obj.checkpoint_dir = None
        # Mock the model_dump method used by CheckpointManager
        mock_config_dict = {'batch_size': 32} # Example dumped dict
        mock_config_obj.model_dump.return_value = mock_config_dict
        
        # Create a minimal trainer instance with the mock config
        trainer = Trainer(
            model=mock_model,
            train_dataloader=mock_dataloader,
            optimizer=mock_optimizer,
            config=mock_config_obj # Pass the mock TrainingConfig object
        )
        
        # Mock the generator instance returned by the __init__ of TextGenerator
        mock_generator_instance = mock_text_generator.return_value
        mock_generator_instance.generate_text.return_value = ["Generated text"]

        prompt = "Test prompt"
        max_new = 50
        temp = 0.7
        
        # --- Call Method ---
        result = trainer.generate_text(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temp,
            # Pass other args if needed for testing
        )

        # --- Assertions ---
        # Check TextGenerator was initialized correctly
        mock_text_generator.assert_called_once_with(
            model=mock_model,
            device=mock_cpu_device,
            config=mock_config_obj # Add config to expected call
        )

        # Check the generate_text method on the *instance* was called
        mock_generator_instance.generate_text.assert_called_once_with(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temp,
            # Add other expected default args based on TextGenerator defaults
            top_k=40,          # Default from TextGenerator
            top_p=0.9,         # Default from TextGenerator
            repetition_penalty=1.0, # Default from TextGenerator
            num_return_sequences=1, # Default from Trainer.generate_text
            use_beam_search=False, # Default from TextGenerator
            num_beams=5,           # Default from TextGenerator
            length_penalty=1.0,    # Default from TextGenerator
            early_stopping=True    # Default from TextGenerator
        )

        assert result == ["Generated text"] 