"""
Tests for the SampleGenerationCallback.
"""
import pytest
import torch
from unittest.mock import MagicMock, call, patch

from craft.training.callbacks import SampleGenerationCallback

# --- Mocks & Fixtures --- #

# mock_trainer, mock_tokenizer, mock_model_with_generate now provided by conftest.py

@pytest.fixture
def sample_callback(mock_tokenizer): # mock_tokenizer comes from conftest
    """Creates a SampleGenerationCallback instance."""
    return SampleGenerationCallback(
        tokenizer=mock_tokenizer,
        prompt="Once upon a time",
        sample_every_n_steps=5,
        sample_on_epoch_end=True,
        max_new_tokens=10,
        temperature=0.8,
        top_k=40,
        num_samples=1
    )

# --- Test Class --- #

class TestSampleGenerationCallback:

    def test_init(self, sample_callback, mock_tokenizer):
        """Test initialization of SampleGenerationCallback."""
        assert sample_callback.tokenizer == mock_tokenizer
        assert sample_callback.prompt == "Once upon a time"
        assert sample_callback.sample_every_n_steps == 5
        assert sample_callback.sample_on_epoch_end is True
        assert sample_callback.max_new_tokens == 10
        assert sample_callback.temperature == 0.8
        assert sample_callback.top_k == 40
        assert sample_callback.num_samples == 1
        assert sample_callback.device is None # Device set later

    def test_set_trainer(self, sample_callback, mock_trainer):
        """Test setting the trainer and deriving the device."""
        sample_callback.set_trainer(mock_trainer)
        assert sample_callback.trainer == mock_trainer
        assert sample_callback.device == mock_trainer.device

    def test_generate_samples_called_on_step_end(self, sample_callback, mock_trainer):
        """Test that _generate_samples is called at correct step intervals."""
        sample_callback.set_trainer(mock_trainer)
        sample_callback._generate_samples = MagicMock() # Mock the generation method

        # Should not trigger on step 4
        sample_callback.on_step_end(step=4)
        sample_callback._generate_samples.assert_not_called()

        # Should trigger on step 5
        sample_callback.on_step_end(step=5)
        sample_callback._generate_samples.assert_called_once_with("Step 5")

        # Should not trigger on step 9
        sample_callback.on_step_end(step=9)
        sample_callback._generate_samples.assert_called_once() # Still called only once

        # Should trigger on step 10
        sample_callback.on_step_end(step=10)
        assert sample_callback._generate_samples.call_count == 2
        sample_callback._generate_samples.assert_called_with("Step 10")

    def test_generate_samples_called_on_epoch_end(self, sample_callback, mock_trainer):
        """Test that _generate_samples is called on epoch end if configured."""
        sample_callback.set_trainer(mock_trainer)
        sample_callback._generate_samples = MagicMock() # Mock the generation method

        sample_callback.on_epoch_end(epoch=0)
        sample_callback._generate_samples.assert_called_once_with("Epoch 1 End")

        sample_callback.on_epoch_end(epoch=1)
        assert sample_callback._generate_samples.call_count == 2
        sample_callback._generate_samples.assert_called_with("Epoch 2 End")

    def test_generate_samples_not_called_if_disabled(self, mock_tokenizer, mock_trainer):
        """Test that _generate_samples is not called if disabled."""
        # Disable both step and epoch generation
        callback = SampleGenerationCallback(tokenizer=mock_tokenizer, sample_every_n_steps=0, sample_on_epoch_end=False)
        callback.set_trainer(mock_trainer)
        callback._generate_samples = MagicMock()

        callback.on_step_end(step=100)
        callback.on_epoch_end(epoch=5)

        callback._generate_samples.assert_not_called()

    def test_generate_samples_logic(self, sample_callback, mock_trainer, mock_tokenizer, mock_model_with_generate):
        """Test the internal logic of the _generate_samples method."""
        # Assign the mock model *with* generate to the mock trainer
        mock_trainer.model = mock_model_with_generate # Assign the specific model needed
        sample_callback.set_trainer(mock_trainer)

        # Call the actual method (indirectly via on_step_end)
        sample_callback.on_step_end(step=5)

        # 1. Check model mode changes
        mock_model_with_generate.eval.assert_called_once()
        mock_model_with_generate.train.assert_called_once()
        # Ensure eval is called before train
        assert mock_model_with_generate.method_calls.index(call.eval()) < mock_model_with_generate.method_calls.index(call.train())

        # 2. Check tokenizer calls
        # Tokenizer __call__ for encoding
        mock_tokenizer.assert_called_once_with("Once upon a time", return_tensors="pt")
        # Tokenizer decode
        # Need expected input tensor for decode (output minus prompt)
        expected_decode_input = torch.tensor([1, 2, 3, 4, 5]) # Generated tokens only
        # Use assert_called_with for the last call if others are possible
        # Check the *last* call to decode, allowing for potential other calls if setup changes
        # Ensure decode was actually called before accessing mock_calls
        mock_tokenizer.decode.assert_called()
        # Unpack name (unused), positional args, and keyword args
        _, decode_pos_args, decode_kwargs = mock_tokenizer.decode.mock_calls[-1]
        assert torch.equal(decode_pos_args[0], expected_decode_input)
        assert decode_kwargs.get('skip_special_tokens') is True

        # 3. Check model.generate call
        # Extract expected input_ids and attention_mask from the mock tokenizer's side effect
        encoded_prompt = mock_tokenizer.side_effect("Once upon a time", return_tensors="pt") # Call the side_effect directly
        expected_input_ids = encoded_prompt['input_ids'].to(mock_trainer.device)
        expected_attention_mask = encoded_prompt['attention_mask'].to(mock_trainer.device)

        mock_model_with_generate.generate.assert_called_once()
        call_args = mock_model_with_generate.generate.call_args[1] # Get keyword args
        assert torch.equal(call_args['input_ids'], expected_input_ids)
        assert torch.equal(call_args['attention_mask'], expected_attention_mask)
        assert call_args['max_new_tokens'] == 10
        assert call_args['temperature'] == 0.8
        assert call_args['top_k'] == 40
        assert call_args['num_return_sequences'] == 1
        assert call_args['do_sample'] is True
        assert call_args['pad_token_id'] == mock_tokenizer.eos_token_id

    # --- Edge Cases for SampleGenerationCallback ---

    def test_init_no_prompt(self, mock_tokenizer):
        """Test initialization with prompt=None."""
        callback = SampleGenerationCallback(tokenizer=mock_tokenizer, prompt=None)
        assert callback.prompt is None

    def test_generate_sample_no_tokenizer_or_trainer(self, sample_callback, mock_trainer):
        """Test _generate_samples logs error if tokenizer or trainer missing."""
        # Patch the callback's logger
        with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
            # Scenario 1: No trainer (and therefore no device)
            sample_callback.trainer = None
            sample_callback.device = None
            sample_callback._generate_samples("Test Trigger")
            mock_logger.error.assert_called_with("Trainer or tokenizer not set properly.")
            # Reset for next scenario
            mock_logger.reset_mock()
            # Restore trainer but remove tokenizer
            sample_callback.trainer = mock_trainer # Restore mock trainer
            sample_callback.device = mock_trainer.device
            sample_callback.tokenizer = None
            sample_callback._generate_samples("Test Trigger")
            mock_logger.error.assert_called_with("Trainer or tokenizer not set properly.")

    def test_generate_sample_model_error(self, sample_callback, mock_trainer, mock_model_with_generate):
        """Test _generate_samples logs error if model.generate fails."""
        # Assign the mock model *with* generate to the mock trainer
        mock_trainer.model = mock_model_with_generate
        sample_callback.set_trainer(mock_trainer)
        # Mock model.generate to raise an error
        mock_model_with_generate.generate.side_effect = Exception("Generation failed!")

        # Patch the callback's logger
        with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
            # Call the generation method (implicitly via step end)
            sample_callback.on_step_end(step=5)

            # Assert error was logged
            mock_logger.error.assert_called_once()
            # Check that the log message contains the exception details
            log_message = mock_logger.error.call_args[0][0]
            assert "Error during sample generation" in log_message
            assert "Generation failed!" in log_message
            # Ensure model was still switched back to train mode
            mock_model_with_generate.train.assert_called_once()

    def test_generate_sample_model_missing_generate(self, sample_callback, mock_trainer):
        """Test _generate_samples warning if model has no generate method."""
        # Create a mock model *without* a generate method (use basic mock_model from conftest)
        mock_model_no_generate = MagicMock(spec=['eval', 'train']) # No generate
        mock_trainer.model = mock_model_no_generate # Assign this model to trainer
        sample_callback.set_trainer(mock_trainer)

        # Patch the callback's logger
        with patch.object(sample_callback, 'logger', MagicMock()) as mock_logger:
            sample_callback._generate_samples("Test Trigger")

            # Assert warning was logged and method returned early
            mock_logger.warning.assert_called_once()
            log_message = mock_logger.warning.call_args[0][0]
            assert "does not have a callable 'generate' method" in log_message
            # Ensure eval/train were not called because it returned early
            mock_model_no_generate.eval.assert_not_called()
            mock_model_no_generate.train.assert_not_called() 