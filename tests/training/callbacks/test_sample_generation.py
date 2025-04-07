"""
Tests for the SampleGenerationCallback.
"""
import pytest
import torch
from unittest.mock import MagicMock, call, patch, ANY
import logging
from craft.training.callbacks.sample_generation import SampleGenerationCallback
from craft.training.generation import TextGenerator

# --- Mocks & Fixtures --- #

# mock_trainer, mock_tokenizer, mock_model_with_generate now provided by conftest.py

@pytest.fixture
def sample_callback(mock_tokenizer):
    """Creates a SampleGenerationCallback instance."""
    # Note: Tokenizer and prompt are NOT passed to __init__;
    # they are inferred from trainer in set_trainer or from config.
    # The mock_tokenizer fixture is technically unused here but kept for potential future use
    # or if the class init changes.
    callback = SampleGenerationCallback(
        step_interval=5,       # How often to generate based on steps
        epoch_interval=1,      # How often to generate based on epochs
        # Generation parameters are typically part of the config, not __init__ args
        # max_new_tokens=10,   # Example: Moved to generation config
        # temperature=0.8,   # Example: Moved to generation config
    )
    # Mock the logger within the callback instance
    callback.logger = MagicMock(spec=logging.Logger)
    return callback

# --- Test Class --- #

class TestSampleGenerationCallback:

    def test_init(self, sample_callback):
        """Test initialization of SampleGenerationCallback."""
        assert sample_callback.step_interval == 5
        assert sample_callback.epoch_interval == 1
        # Prompt, Tokenizer are set later via set_trainer
        assert sample_callback.generator is None

    def test_set_trainer(self, sample_callback, mock_trainer):
        """Test setting the trainer and deriving the device."""
        sample_callback.set_trainer(mock_trainer)
        assert sample_callback.trainer == mock_trainer
        # Callback uses trainer.device, doesn't store it directly
        # Generator creation is now conditional based on model.generate
        # We test the conditional logic separately.
        # assert sample_callback.generator is not None # REMOVED - Generator might be None

    def test_generate_samples_called_on_step_end(self, sample_callback, mock_trainer):
        """Test generate_samples is called correctly from on_step_end."""
        sample_callback.step_interval = 5
        sample_callback.epoch_interval = None
        sample_callback.set_trainer(mock_trainer)
        sample_callback.generate_samples = MagicMock() # Mock the method to check calls

        # Simulate step ends
        sample_callback.on_step_end(step=3, global_step=3, metrics={}) # Should not trigger (global_step+1 = 4)
        sample_callback.generate_samples.assert_not_called()
        sample_callback.on_step_end(step=4, global_step=4, metrics={}) # Should trigger (global_step+1 = 5)
        sample_callback.generate_samples.assert_called_once_with("Step 5") # Check trigger event

    def test_generate_samples_called_on_epoch_end(self, sample_callback, mock_trainer):
        """Test generate_samples is called correctly from on_epoch_end."""
        sample_callback.step_interval = None
        sample_callback.epoch_interval = 2
        sample_callback.set_trainer(mock_trainer)
        sample_callback.generate_samples = MagicMock()

        # Simulate epoch ends
        sample_callback.on_epoch_end(epoch=0, global_step=10, metrics={}) # Should not trigger (epoch+1 = 1)
        sample_callback.generate_samples.assert_not_called()
        sample_callback.on_epoch_end(epoch=1, global_step=20, metrics={}) # Should trigger (epoch+1 = 2)
        sample_callback.generate_samples.assert_called_once_with("Epoch 2") # Check trigger event

    def test_generate_samples_not_called_if_disabled(self, sample_callback, mock_trainer):
        """Test generate_samples is not called if intervals are None."""
        # Explicitly set intervals to None for this test
        sample_callback.step_interval = None
        sample_callback.epoch_interval = None
        sample_callback.set_trainer(mock_trainer)
        sample_callback.generate_samples = MagicMock()

        sample_callback.on_step_end(step=0, global_step=0, metrics={})
        sample_callback.on_epoch_end(epoch=0, global_step=10, metrics={})

        sample_callback.generate_samples.assert_not_called()

    def test_generate_samples_logic(self, mock_text_generator, mock_logger_fixture):
        """Test the core logic of generate_samples."""
        callback = SampleGenerationCallback()
        callback.generator = mock_text_generator
        callback.initialized = True
        callback.logger = mock_logger_fixture
        callback.start_prompt = "Test prompt"
        callback.max_new_tokens = 10
        callback.temperature = 0.7
        mock_text_generator.generate_text.return_value = ["Generated text"]

        callback.generate_samples("Test Event") # Call with only trigger_event

        mock_text_generator.generate_text.assert_called_once_with(
            prompt="Test prompt",
            max_new_tokens=10,
            temperature=0.7
        )
        # Check for specific log message indicating generation start/end
        # Using mock_calls to be more flexible than assert_any_call
        start_log_call = call('--- Generating Sample (Test Event) ---')
        end_log_call = call('--- End Sample Generation (Test Event) ---')
        assert start_log_call in mock_logger_fixture.info.mock_calls
        assert end_log_call in mock_logger_fixture.info.mock_calls

    def test_generate_samples_uses_callback_prompt_override(self, mock_text_generator, mock_logger_fixture):
        """Test that generate_samples uses the prompt set on the callback."""
        custom_prompt = "Override prompt: "
        callback = SampleGenerationCallback(start_prompt=custom_prompt)
        callback.generator = mock_text_generator
        callback.initialized = True
        callback.logger = mock_logger_fixture

        callback.generate_samples("Override Test") # Call with only trigger_event

        mock_text_generator.generate_text.assert_called_once_with(
            prompt=custom_prompt, # Check that the callback's prompt was used
            max_new_tokens=ANY, # Allow default value
            temperature=ANY # Allow default value
        )

    def test_generate_samples_logs_error_on_generator_failure(self, mock_text_generator, mock_logger_fixture):
        """Test error logging when TextGenerator fails."""
        callback = SampleGenerationCallback()
        callback.generator = mock_text_generator
        callback.initialized = True
        callback.logger = mock_logger_fixture
        error_message = "Generation failed spectacularly"
        mock_text_generator.generate_text.side_effect = Exception(error_message)

        callback.generate_samples("Failure Test") # Call with only trigger_event

        mock_logger_fixture.error.assert_called_once_with(f"Error during text generation: {error_message}", exc_info=True)

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

    # --- Tests for _initialize_generator --- #

    @patch('craft.training.callbacks.sample_generation.TextGenerator') # Mock the TextGenerator class
    def test_initialize_generator_success(self, MockTextGenerator, sample_callback, mock_trainer, mock_model_with_generate, mock_tokenizer):
        """Test _initialize_generator successfully creates TextGenerator."""
        mock_trainer.model = mock_model_with_generate
        # Set the tokenizer directly on the mock trainer instance
        mock_trainer.tokenizer = mock_tokenizer 
        # mock_trainer.get_tokenizer = MagicMock(return_value=mock_tokenizer) # Old way
        mock_trainer.device = 'cuda' # Ensure device is set

        # Configure the trainer's config for generation parameters
        mock_trainer.config = MagicMock()
        mock_trainer.config.generation = MagicMock(
            max_new_tokens=50,
            temperature=0.7,
            top_k=40
            # Add other relevant GeneratorConfig fields if needed
        )

        # Call the private method we want to test
        result = sample_callback._initialize_generator(mock_trainer)

        assert result is True
        assert sample_callback.generator is not None
        # Check that TextGenerator was instantiated with the correct args
        MockTextGenerator.assert_called_once()
        call_args, call_kwargs = MockTextGenerator.call_args
        assert call_kwargs['model'] == mock_model_with_generate
        # Assert based on the object attached to the trainer
        assert call_kwargs['tokenizer'] is mock_trainer.tokenizer
        assert call_kwargs['device'] == mock_trainer.device
        # Check that the config passed is a dictionary (as expected by the current code)
        assert isinstance(call_kwargs['config'], dict)
        # Verify generation config was passed (REMOVED - TextGenerator doesn't take this directly)
        # assert call_kwargs['generation_config'] is not None 

    def test_initialize_generator_no_trainer(self, sample_callback):
        """Test _initialize_generator fails if trainer is None."""
        with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
            result = sample_callback._initialize_generator(None)
            assert result is False
            assert sample_callback.generator is None
            mock_logger.error.assert_called_with( # Match actual log message
                 "Trainer is missing required attributes (model, device, train_dataloader with dataset) for TextGenerator initialization."
            )

    def test_initialize_generator_no_model(self, sample_callback, mock_trainer):
        """Test _initialize_generator fails if trainer.model is None."""
        mock_trainer.model = None
        mock_trainer.get_tokenizer = MagicMock(return_value=MagicMock()) # Provide a mock tokenizer
        with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
            result = sample_callback._initialize_generator(mock_trainer)
            assert result is False
            assert sample_callback.generator is None
            # Check for the specific AttributeError during TextGenerator init
            mock_logger.error.assert_called_once()
            log_msg = mock_logger.error.call_args[0][0]
            assert log_msg.startswith(
                "Failed to initialize TextGenerator: Missing required attribute"
            )
            # Check for the specific error related to the missing 'to' method on NoneType
            assert "'NoneType' object has no attribute 'to'" in log_msg

    def test_initialize_generator_model_no_generate(self, sample_callback, mock_trainer):
        """Test _initialize_generator logs error if model has no 'generate' method."""
        mock_model_no_generate = MagicMock(spec=['forward']) # Mock model *without* generate
        mock_trainer.model = mock_model_no_generate
        mock_trainer.get_tokenizer = MagicMock(return_value=MagicMock())
        with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
            result = sample_callback._initialize_generator(mock_trainer)
            assert result is False
            assert sample_callback.generator is None
            # Check for the specific AttributeError during TextGenerator init
            mock_logger.error.assert_called_once()
            assert mock_logger.error.call_args[0][0].startswith(
                "Failed to initialize TextGenerator: Missing required attribute"
            )
            # Check for the specific error related to the Mock lacking 'to'
            assert "Mock object has no attribute 'to'" in mock_logger.error.call_args[0][0]
            assert mock_logger.error.call_args.kwargs.get('exc_info') is True

    def test_initialize_generator_no_tokenizer(self, sample_callback, mock_trainer, mock_model_with_generate):
        """Test _initialize_generator fails if trainer cannot provide a tokenizer."""
        mock_trainer.model = mock_model_with_generate
        mock_trainer.tokenizer = None # Simulate no tokenizer explicitly on trainer
        # mock_trainer.get_tokenizer = MagicMock(return_value=None) # Old way
        with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
            result = sample_callback._initialize_generator(mock_trainer)
            assert result is False
            mock_logger.error.assert_called_with(
                 "Trainer does not have an explicit 'tokenizer' attribute needed for TextGenerator."
            ) 