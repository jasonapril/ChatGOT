import pytest
from unittest.mock import MagicMock, call, patch
from typing import List
import torch

# Import the classes to test
from craft.training.callbacks import Callback, CallbackList

# --- Mocks --- #

@pytest.fixture
def mock_callback():
    """Provides a basic MagicMock callback."""
    # Create a mock that adheres to the Callback interface (optional but good practice)
    callback = MagicMock(spec=Callback)
    return callback

@pytest.fixture
def mock_trainer(mock_model_with_generate):
    """Creates a mock trainer object with optimizer, model, and device."""
    trainer = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{'lr': 0.01}]
    trainer.model = mock_model_with_generate # Assign the mock model
    trainer.device = torch.device("cpu")     # Define a device
    return trainer

# --- Tests for CallbackList --- #

class TestCallbackList:

    def test_init_empty(self):
        """Test initialization with no callbacks."""
        cb_list = CallbackList()
        assert cb_list.callbacks == []

    def test_init_with_callbacks(self, mock_callback):
        """Test initialization with a list of callbacks."""
        callbacks = [mock_callback, MagicMock(spec=Callback)]
        cb_list = CallbackList(callbacks)
        assert cb_list.callbacks == callbacks

    def test_append(self, mock_callback):
        """Test appending a single callback."""
        cb_list = CallbackList()
        cb_list.append(mock_callback)
        assert cb_list.callbacks == [mock_callback]

    def test_extend(self, mock_callback):
        """Test extending with multiple callbacks."""
        cb_list = CallbackList([mock_callback])
        new_callbacks = [MagicMock(spec=Callback), MagicMock(spec=Callback)]
        cb_list.extend(new_callbacks)
        assert cb_list.callbacks == [mock_callback] + new_callbacks

    def test_set_trainer(self, mock_trainer):
        """Test that set_trainer calls set_trainer on all contained callbacks."""
        mock_cb1 = MagicMock(spec=Callback)
        mock_cb2 = MagicMock(spec=Callback)
        cb_list = CallbackList([mock_cb1, mock_cb2])

        cb_list.set_trainer(mock_trainer)

        mock_cb1.set_trainer.assert_called_once_with(mock_trainer)
        mock_cb2.set_trainer.assert_called_once_with(mock_trainer)

    # Test dispatching of each event type
    @pytest.mark.parametrize(
        "method_name, method_args",
        [
            ("on_train_begin", {}),
            ("on_train_end", {"logs": {"final_metric": 1}}),
            ("on_epoch_begin", {"epoch": 1, "logs": {"epoch_start": True}}),
            ("on_epoch_end", {"epoch": 1, "logs": {"epoch_loss": 0.5}}),
            ("on_step_begin", {"step": 100, "logs": {}}),
            ("on_step_end", {"step": 100, "logs": {"loss": 0.1}}),
        ]
    )
    def test_event_dispatch(self, method_name, method_args):
        """Test that event methods are correctly dispatched to all callbacks."""
        mock_cb1 = MagicMock(spec=Callback)
        mock_cb2 = MagicMock(spec=Callback)
        cb_list = CallbackList([mock_cb1, mock_cb2])

        # Get the method on CallbackList and call it
        list_method = getattr(cb_list, method_name)
        list_method(**method_args)

        # Assert the corresponding method was called on each mock callback
        # Separate assertions for clarity
        cb1_method = getattr(mock_cb1, method_name)
        cb2_method = getattr(mock_cb2, method_name)

        # Unpack args based on method name for assertion
        pos_args = []
        kw_args = {}
        if "epoch" in method_args: pos_args.append(method_args["epoch"])
        if "step" in method_args: pos_args.append(method_args["step"])
        # Ensure logs dict is passed, even if originally None/empty
        kw_args["logs"] = method_args.get("logs", {})

        cb1_method.assert_called_once_with(*pos_args, **kw_args)
        cb2_method.assert_called_once_with(*pos_args, **kw_args)


# --- Tests for ReduceLROnPlateauOrInstability --- #

class TestReduceLROnPlateauOrInstability:
    @pytest.fixture
    def lr_callback(self):
        """Creates a ReduceLROnPlateauOrInstability callback instance."""
        from craft.training.callbacks import ReduceLROnPlateauOrInstability
        return ReduceLROnPlateauOrInstability(
            monitor='loss',
            factor=0.5,
            patience=2,
            min_lr=0.0001,
            threshold=0.1,
            cooldown=1,
            window_size=5,  # Smaller window for testing
            verbose=False   # Disable logging in tests
        )

    def test_init(self, lr_callback):
        """Test initialization with default parameters."""
        assert lr_callback.monitor == 'loss'
        assert lr_callback.factor == 0.5
        assert lr_callback.patience == 2
        assert lr_callback.min_lr == 0.0001
        assert lr_callback.threshold == 0.1
        assert lr_callback.cooldown == 1
        assert lr_callback.window_size == 5
        assert lr_callback.verbose is False
        assert lr_callback.wait == 0
        assert lr_callback.cooldown_counter == 0
        assert lr_callback.best_loss == float('inf')
        assert lr_callback.recent_losses == []

    def test_set_trainer(self, lr_callback, mock_trainer):
        """Test setting the trainer and initializing state."""
        lr_callback.set_trainer(mock_trainer)
        lr_callback.on_train_begin()  # This will initialize initial_lr
        assert lr_callback.trainer == mock_trainer
        assert lr_callback.optimizer == mock_trainer.optimizer
        assert lr_callback.initial_lr == 0.01  # From mock_trainer fixture

    def test_on_step_end_plateau(self, lr_callback, mock_trainer):
        """Test learning rate reduction on plateau."""
        lr_callback.set_trainer(mock_trainer)
        initial_lr = mock_trainer.optimizer.param_groups[0]['lr']

        # Fill the window with high loss values
        for step in range(5):  # window_size
            lr_callback.on_step_end(step=step, logs={'loss': 0.5})

        # Simulate plateau: loss not improving for patience+1 steps
        for step in range(5, 8):  # patience + 1
            lr_callback.on_step_end(step=step, logs={'loss': 0.5})

        # Check if learning rate was reduced
        assert mock_trainer.optimizer.param_groups[0]['lr'] == initial_lr * 0.5

    def test_on_step_end_instability(self, lr_callback, mock_trainer):
        """Test learning rate reduction on instability."""
        lr_callback.set_trainer(mock_trainer)
        initial_lr = mock_trainer.optimizer.param_groups[0]['lr']

        # Fill window with good loss values
        for step in range(5):  # window_size
            lr_callback.on_step_end(step=step, logs={'loss': 0.1})

        # Simulate instability: loss suddenly increases significantly
        for step in range(5, 8):  # patience + 1
            lr_callback.on_step_end(step=step, logs={'loss': 1.0})  # 10x increase

        # Check if learning rate was reduced
        assert mock_trainer.optimizer.param_groups[0]['lr'] == initial_lr * 0.5

    def test_on_step_end_cooldown(self, lr_callback, mock_trainer):
        """Test cooldown period after learning rate reduction."""
        lr_callback.set_trainer(mock_trainer)
        initial_lr = mock_trainer.optimizer.param_groups[0]['lr']

        # Fill window and trigger reduction
        for step in range(5):  # window_size
            lr_callback.on_step_end(step=step, logs={'loss': 0.5})
        for step in range(5, 8):  # patience + 1
            lr_callback.on_step_end(step=step, logs={'loss': 0.5})

        # During cooldown, learning rate should not change even if loss is bad
        for step in range(8, 10):
            lr_callback.on_step_end(step=step, logs={'loss': 1.0})
        assert mock_trainer.optimizer.param_groups[0]['lr'] == initial_lr * 0.5

    def test_on_step_end_min_lr(self, lr_callback, mock_trainer):
        """Test that learning rate doesn't go below min_lr."""
        lr_callback.set_trainer(mock_trainer)
        initial_lr = mock_trainer.optimizer.param_groups[0]['lr']

        # Trigger multiple learning rate reductions
        for i in range(5):  # More than enough to go below min_lr
            # Fill window
            for step in range(i * 10, i * 10 + 5):
                lr_callback.on_step_end(step=step, logs={'loss': 0.5})
            # Trigger reduction
            for step in range(i * 10 + 5, i * 10 + 8):
                lr_callback.on_step_end(step=step, logs={'loss': 0.5})

        # Check that learning rate didn't go below min_lr
        assert mock_trainer.optimizer.param_groups[0]['lr'] >= lr_callback.min_lr

# --- Tests for SampleGenerationCallback --- #

@pytest.fixture
def mock_tokenizer():
    """Creates a mock tokenizer with encode/decode methods."""
    tokenizer = MagicMock()
    # Simulate HF tokenizer __call__ behavior
    def encode_side_effect(text, return_tensors=None):
        # Dummy encoding, return simple tensor-like structure
        if text == "Once upon a time":
            input_ids = torch.tensor([[101, 5141, 2747, 1037, 1994, 102]])
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        return {'input_ids': torch.tensor([[0]]), 'attention_mask': torch.tensor([[1]])}
    # Use side_effect to make the mock callable like a function that returns the dict
    tokenizer.side_effect = encode_side_effect 
    tokenizer.decode.return_value = " generated sample text"
    tokenizer.eos_token_id = 2 # Dummy EOS token id
    return tokenizer

@pytest.fixture
def mock_model_with_generate():
    """Creates a mock model with a callable generate method."""
    model = MagicMock()
    # Simulate generate output (needs to include prompt tokens based on callback logic)
    # Prompt: [101, 5141, 2747, 1037, 1994, 102] -> 6 tokens
    # Generate 5 new tokens: [1, 2, 3, 4, 5]
    generated_ids = torch.tensor([[101, 5141, 2747, 1037, 1994, 102, 1, 2, 3, 4, 5]])
    model.generate.return_value = generated_ids
    model.eval = MagicMock()
    model.train = MagicMock()
    return model

@pytest.fixture
def sample_callback(mock_tokenizer):
    """Creates a SampleGenerationCallback instance."""
    from craft.training.callbacks import SampleGenerationCallback
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
        from craft.training.callbacks import SampleGenerationCallback
        # Disable both step and epoch generation
        callback = SampleGenerationCallback(tokenizer=mock_tokenizer, sample_every_n_steps=0, sample_on_epoch_end=False)
        callback.set_trainer(mock_trainer)
        callback._generate_samples = MagicMock()

        callback.on_step_end(step=100)
        callback.on_epoch_end(epoch=5)

        callback._generate_samples.assert_not_called()

    def test_generate_samples_logic(self, sample_callback, mock_trainer, mock_tokenizer, mock_model_with_generate):
        """Test the internal logic of the _generate_samples method."""
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

# --- Tests for TensorBoardLogger --- #

@pytest.fixture
def tb_logger_callback(tmp_path): # Use pytest's tmp_path fixture for log_dir
    """Creates a TensorBoardLogger instance."""
    from craft.training.callbacks import TensorBoardLogger
    log_dir = tmp_path / "tb_logs"
    return TensorBoardLogger(log_dir=str(log_dir))

class TestTensorBoardLogger:

    def test_init(self, tb_logger_callback, tmp_path):
        """Test TensorBoardLogger initialization."""
        expected_log_dir = str(tmp_path / "tb_logs")
        assert tb_logger_callback.log_dir == expected_log_dir
        assert tb_logger_callback.writer is None

    def test_on_train_begin_initializes_writer(self, tb_logger_callback):
        """Test that SummaryWriter is initialized on train begin."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_class.assert_called_once_with(tb_logger_callback.log_dir)
            # Check that the instance is stored
            assert tb_logger_callback.writer == mock_writer_class.return_value

    def test_on_train_begin_handles_exception(self, tb_logger_callback, caplog):
        """Test that an exception during SummaryWriter init is handled."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            mock_writer_class.side_effect = Exception("Initialization failed")
            tb_logger_callback.on_train_begin()
            assert "Failed to initialize TensorBoard SummaryWriter" in caplog.text
            assert tb_logger_callback.writer is None

    def test_on_step_end_logs_metrics(self, tb_logger_callback):
        """Test logging of step-level metrics."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            # Need to initialize the writer first
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            step = 100
            logs = {'loss': 0.123, 'lr': 0.001, 'other_metric': 5}
            tb_logger_callback.on_step_end(step=step, logs=logs)

            # Check calls to add_scalar
            calls = [
                call.add_scalar('Loss/train_step', 0.123, step),
                call.add_scalar('LearningRate/step', 0.001, step)
                # 'other_metric' is not explicitly logged, so no call expected
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            # Ensure only expected metrics were logged
            assert mock_writer_instance.add_scalar.call_count == 2

    def test_on_step_end_no_logs_or_writer(self, tb_logger_callback):
        """Test that nothing happens if writer not initialized or logs are None."""
        # Scenario 1: Writer not initialized (No patch needed)
        tb_logger_callback.on_step_end(step=1, logs={'loss': 0.1})
        # add_scalar should not be available or called (writer is None)

        # Scenario 2: Writer initialized, logs is None
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value
            tb_logger_callback.on_step_end(step=1, logs=None)
            mock_writer_instance.add_scalar.assert_not_called()

    def test_on_epoch_end_logs_metrics(self, tb_logger_callback):
        """Test logging of epoch-level metrics."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            epoch = 5
            logs = {'loss': 0.5, 'val_loss': 0.6, 'perplexity': 10.0}
            tb_logger_callback.on_epoch_end(epoch=epoch, logs=logs)

            calls = [
                call.add_scalar('Loss/train_epoch', 0.5, epoch + 1),
                call.add_scalar('Loss/validation_epoch', 0.6, epoch + 1)
                # 'perplexity' is not explicitly logged
            ]
            mock_writer_instance.assert_has_calls(calls, any_order=True)
            assert mock_writer_instance.add_scalar.call_count == 2

    def test_on_train_end_closes_writer(self, tb_logger_callback):
        """Test that the writer is closed on train end."""
        with patch('craft.training.callbacks.SummaryWriter', autospec=True) as mock_writer_class:
            tb_logger_callback.on_train_begin()
            mock_writer_instance = mock_writer_class.return_value

            tb_logger_callback.on_train_end()
            mock_writer_instance.close.assert_called_once()

    def test_on_train_end_no_writer(self, tb_logger_callback):
        """Test that close is not called if writer wasn't initialized."""
        # Writer is None initially (No patch needed)
        tb_logger_callback.on_train_end()
        # No error should occur, and close shouldn't be called on None

    # Add simple tests to ensure other required methods exist
    def test_other_methods_exist(self, tb_logger_callback):
        """Check that other abstract methods are implemented (even if empty)."""
        assert hasattr(tb_logger_callback, 'on_epoch_begin')
        assert hasattr(tb_logger_callback, 'on_step_begin')
        # set_trainer is inherited
        assert hasattr(tb_logger_callback, 'set_trainer') 