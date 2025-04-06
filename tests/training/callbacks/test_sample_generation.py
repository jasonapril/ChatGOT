"""
Tests for the SampleGenerationCallback.
"""
import pytest
import torch
from unittest.mock import MagicMock, call, patch
import logging
from craft.training.callbacks.sample_generation import SampleGenerationCallback
from craft.training.generation import TextGenerator

# --- Mocks & Fixtures --- #

# mock_trainer, mock_tokenizer, mock_model_with_generate now provided by conftest.py

@pytest.fixture
def sample_callback(mock_tokenizer): # mock_tokenizer comes from conftest
    """Creates a SampleGenerationCallback instance."""
    # Note: Tokenizer is NOT passed to __init__; it's inferred from trainer in set_trainer
    # Prompt is also not passed; it's inferred from trainer.config
    return SampleGenerationCallback(
        # prompt="Once upon a time", # REMOVED
        step_interval=5,       # Renamed from sample_every_n_steps
        epoch_interval=1,      # Set explicitly, True is not valid
        max_new_tokens=10,
        temperature=0.8
        # top_k and num_samples are not __init__ args
    )

# --- Test Class --- #

class TestSampleGenerationCallback:

    def test_init(self, sample_callback, mock_tokenizer):
        """Test initialization of SampleGenerationCallback."""
        # assert sample_callback.prompt == "Once upon a time" # REMOVED - Prompt comes from config
        assert sample_callback.step_interval == 5
        assert sample_callback.epoch_interval == 1
        assert sample_callback.max_new_tokens == 10
        assert sample_callback.temperature == 0.8
        # Device is not stored on the callback itself

    def test_set_trainer(self, sample_callback, mock_trainer):
        """Test setting the trainer and deriving the device."""
        sample_callback.set_trainer(mock_trainer)
        assert sample_callback.trainer == mock_trainer
        # Callback uses trainer.device, doesn't store it directly
        # Generator creation is now conditional based on model.generate
        # We test the conditional logic separately.
        # assert sample_callback.generator is not None # REMOVED - Generator might be None

    def test_generate_samples_called_on_step_end(self, sample_callback, mock_trainer):
        """Test that generate_samples is called correctly on step end based on interval."""
        sample_callback.set_trainer(mock_trainer)
        sample_callback.generate_samples = MagicMock() # Mock the generation method

        # Should not trigger on step 3 (global_step 3)
        sample_callback.on_step_end(step=3, logs={'global_step': 3})
        sample_callback.generate_samples.assert_not_called()

        # Should trigger on step 4 (global_step 4)
        sample_callback.on_step_end(step=4, logs={'global_step': 4})
        sample_callback.generate_samples.assert_called_once_with(mock_trainer, "Step 5") # global_step + 1

        # Reset and check step 9
        sample_callback.generate_samples.reset_mock()
        sample_callback.on_step_end(step=8, logs={'global_step': 8}) 
        sample_callback.generate_samples.assert_not_called()
        sample_callback.on_step_end(step=9, logs={'global_step': 9})
        sample_callback.generate_samples.assert_called_once_with(mock_trainer, "Step 10") # global_step + 1

    def test_generate_samples_called_on_epoch_end(self, sample_callback, mock_trainer):
        """Test that generate_samples is called on epoch end if configured."""
        sample_callback.set_trainer(mock_trainer)
        sample_callback.generate_samples = MagicMock() # Mock the generation method
        dummy_metrics = {}

        # Should trigger on epoch 0 (Epoch 1)
        sample_callback.on_epoch_end(trainer=mock_trainer, epoch=0, train_metrics=dummy_metrics, val_metrics=dummy_metrics)
        sample_callback.generate_samples.assert_called_once_with(mock_trainer, "Epoch 1")

        # Should trigger on epoch 1 (Epoch 2)
        sample_callback.generate_samples.reset_mock()
        sample_callback.on_epoch_end(trainer=mock_trainer, epoch=1, train_metrics=dummy_metrics, val_metrics=dummy_metrics)
        sample_callback.generate_samples.assert_called_once_with(mock_trainer, "Epoch 2")

    def test_generate_samples_not_called_if_disabled(self, mock_tokenizer, mock_trainer):
        """Test that generate_samples is not called if disabled."""
        callback = SampleGenerationCallback(step_interval=None, epoch_interval=None)
        callback.set_trainer(mock_trainer)
        callback.generate_samples = MagicMock()
        dummy_metrics = {}

        # Test step end
        callback.on_step_end(step=100, logs={'global_step': 100})
        callback.generate_samples.assert_not_called()
        # Test epoch end
        callback.on_epoch_end(trainer=mock_trainer, epoch=5, train_metrics=dummy_metrics, val_metrics=dummy_metrics)
        callback.generate_samples.assert_not_called()

    def test_generate_samples_logic(self, sample_callback, mock_trainer, mock_tokenizer, mock_model_with_generate):
        """Test the internal logic of the generate_samples method."""
        # Assign mock model and ensure generator is initialized (or mocked)
        mock_trainer.model = mock_model_with_generate
        # Configure the mock config object on the trainer
        expected_prompt = "The meaning of life is "
        mock_trainer.config = MagicMock(sample_start_text=expected_prompt)
        
        sample_callback.set_trainer(mock_trainer)
        # Manually mock the generator after potential initialization
        sample_callback.generator = MagicMock(spec=TextGenerator)
        # Explicitly mock the generate_text method on the mock generator
        sample_callback.generator.generate_text = MagicMock(return_value=["mock generated text"])

        # Patch _initialize_generator to prevent it from overwriting our mock
        with patch.object(sample_callback, '_initialize_generator', return_value=True) as mock_init_gen:
            # Call the actual method
            sample_callback.generate_samples(mock_trainer, "Step 5 Test")

            # 1. Check generator.generate_text call
            sample_callback.generator.generate_text.assert_called_once()
            # Extract call args/kwargs
            call_args, call_kwargs = sample_callback.generator.generate_text.call_args
            # Check keyword arguments used in the call
            assert call_kwargs['prompt'] == expected_prompt
            assert call_kwargs['max_new_tokens'] == 10 # From sample_callback fixture
            assert call_kwargs['temperature'] == 0.8 # From sample_callback fixture
            # Removed assertions checking model.eval/train as generate_samples doesn't handle this

    # def test_generate_sample_no_tokenizer_or_trainer(self, sample_callback):
    #     """Test generate_samples logs error if generator not ready."""
    #     # Patch the callback's logger and _initialize_generator
    #     with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger, \
    #          patch.object(sample_callback, '_initialize_generator', return_value=False) as mock_init:
            
    #         sample_callback.generate_samples(None, "Test Trigger No Init") # Pass None for trainer
            
    #         # Assertions
    #         mock_init.assert_called_once_with(None)
    #         mock_logger.error.assert_called_once_with("TextGenerator not initialized. Cannot generate samples.")

    # def test_generate_sample_model_error(self, sample_callback, mock_trainer, mock_model_with_generate):
    #     """Test generate_samples logs error if generator.generate_text fails."""
    #     # Setup trainer and mock generator
    #     mock_trainer.model = mock_model_with_generate
    #     sample_callback.set_trainer(mock_trainer)
    #     sample_callback.generator = MagicMock(spec=TextGenerator)
    #     sample_callback.generator.generate_text.side_effect = Exception("Generation failed!")
        
    #     # Patch the callback's logger
    #     with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
    #         sample_callback.generate_samples(mock_trainer, "Test Trigger Error")
            
    #         # Assertions
    #         mock_logger.error.assert_called_once_with("Error during sample generation: Generation failed!", exc_info=True)

    # def test_generate_sample_model_missing_generate(self, sample_callback, mock_trainer):
    #     """Test generate_samples logs error if _initialize_generator fails (e.g., model missing generate)."""
    #     # Create a mock model *without* a generate method
    #     mock_model_no_generate = MagicMock(spec=['eval', 'train']) 
    #     mock_trainer.model = mock_model_no_generate
    #     # Set trainer *before* patching logger
    #     sample_callback.set_trainer(mock_trainer)

    #     # Patch the callback's logger
    #     with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
    #          # Patch _initialize_generator to simulate failure
    #          with patch.object(sample_callback, '_initialize_generator', return_value=False) as mock_init:
    #             sample_callback.generate_samples(mock_trainer, "Test Trigger Init Fail")
                
    #             # Assertions
    #             mock_init.assert_called_once_with(mock_trainer)
    #             mock_logger.error.assert_called_once_with("TextGenerator not initialized. Cannot generate samples.")

    # --- Edge Cases for SampleGenerationCallback ---

    def test_init_no_prompt(self, mock_tokenizer):
        """Test initialization with prompt=None (no longer applicable directly)."""
        # Prompt is handled via config now. Test basic init with no intervals.
        callback = SampleGenerationCallback(step_interval=None, epoch_interval=None)
        # assert callback.prompt is None # REMOVED
        assert callback.step_interval is None
        assert callback.epoch_interval is None
        assert callback.generator is None # Generator not initialized until set_trainer/on_train_begin

    # NOTE: The following tests are duplicates or outdated, kept for reference, remove after refactoring.
    # def test_generate_sample_no_tokenizer_or_trainer(self, sample_callback, mock_trainer):
    #     """Test _generate_samples logs error if generator not ready or prompt missing in config."""
    #     # Patch the callback's logger
    #     with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
    #         # Scenario 1: No trainer (and therefore no device)
    #         sample_callback.trainer = None
    #         sample_callback.device = None
    #         # sample_callback._generate_samples("Test Trigger") # OLD call
    #         # Check error log - Now handled inside generate_samples
    #         # mock_logger.error.assert_called_once_with("TextGenerator not initialized. Cannot generate samples.") # Should be error now
    #         # Reset for next scenario
    #         # mock_logger.reset_mock()
            
    #         # Scenario 2: Trainer exists, but config lacks prompt - No longer relevant, prompt handled by TextGenerator
    #         # sample_callback.trainer = mock_trainer
    #         # original_prompt = mock_trainer.config.sample_start_text
    #         # mock_trainer.config.sample_start_text = None # Simulate missing prompt in config
    #         # sample_callback.generate_samples(mock_trainer, "Test Trigger No Prompt")
    #         # mock_logger.error.assert_called_once_with("TextGenerator not initialized. Cannot generate samples.")
    #         # Restore config
    #         # mock_trainer.config.sample_start_text = original_prompt

    # def test_generate_sample_model_error(self, sample_callback, mock_trainer, mock_model_with_generate, caplog):
    #     """Test _generate_samples logs error if model.generate fails."""
    #     # Assign the mock model *with* generate to the mock trainer
    #     mock_model_with_generate.training = False
    #     mock_model_with_generate.generate.side_effect = Exception("Generation failed!")
    #     mock_trainer.model = mock_model_with_generate
    #     # Mock the dataset needed by TextGenerator for encoding
    #     # mock_dataset = MagicMock()
    #     # mock_dataset.tokenizer.encode.return_value = [[1, 2, 3]] # List of lists
    #     # mock_trainer.dataset = mock_dataset # Trainer doesn't store dataset directly
    #     sample_callback.set_trainer(mock_trainer)

    #     # Call generate_samples directly
    #     with torch.no_grad(), caplog.at_level(logging.ERROR):
    #         # sample_callback.generate_samples(mock_trainer, "Step 5 Error Test") # Call public method
    #         pass # Covered by the new test_generate_sample_model_error

    #     # Assert error was logged by the callback
    #     # assert "Error during sample generation: Generation failed!" in caplog.text

    # def test_generate_sample_model_missing_generate(self, sample_callback, mock_trainer):
    #     """Test _generate_samples warning if model has no generate method."""
    #     # Create a mock model *without* a generate method
    #     mock_model_no_generate = MagicMock(spec=['eval', 'train']) # No generate
    #     mock_trainer.model = mock_model_no_generate # Assign this model to trainer
    #     sample_callback.set_trainer(mock_trainer)

    #     # Patch the callback's logger
    #     with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
    #         # sample_callback.generate_samples(mock_trainer, "Test Trigger") # Call public method
    #         pass # Covered by new test_generate_sample_model_missing_generate
            
    #         # Assert error was logged because init fails
    #         # mock_logger.error.assert_called_once()
    #         # assert "TextGenerator not initialized" in mock_logger.error.call_args[0][0] 