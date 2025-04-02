import pytest
import torch
import logging
from unittest.mock import MagicMock, patch, PropertyMock

# Class to test
from craft.training.amp import SafeGradScaler

# Mock the base class methods we expect to call via super()
@pytest.fixture
def mock_grad_scaler_base_methods():
    # We patch the methods on the base class itself before the test runs
    patchers = {
        'scale': patch.object(torch.amp.GradScaler, 'scale', return_value=MagicMock(spec=torch.Tensor), autospec=True),
        'step': patch.object(torch.amp.GradScaler, 'step', return_value=None, autospec=True),
        'update': patch.object(torch.amp.GradScaler, 'update', return_value=None, autospec=True),
        'state_dict': patch.object(torch.amp.GradScaler, 'state_dict', return_value={}, autospec=True),
        'load_state_dict': patch.object(torch.amp.GradScaler, 'load_state_dict', return_value=None, autospec=True),
        # Mock _enabled property - needs careful handling if we need to change it
        # Using PropertyMock allows checking getter calls
        # '_enabled': patch.object(torch.amp.GradScaler, '_enabled', new_callable=PropertyMock, return_value=True)
    }
    mocks = {name: p.start() for name, p in patchers.items()}
    yield mocks
    for p in patchers.values():
        p.stop()

@pytest.fixture
def mock_optimizer():
    opt = MagicMock(spec=torch.optim.Optimizer)
    opt.step = MagicMock()
    opt.zero_grad = MagicMock()
    return opt

@pytest.fixture
def mock_logger():
    with patch('craft.training.amp.logger') as logger_mock:
        yield logger_mock

# --- Test Class ---

class TestSafeGradScaler:

    # --- Initialization Tests ---

    def test_init_defaults(self, mock_grad_scaler_base_methods):
        """Test init with default settings."""
        # Note: We pass enabled=True explicitly because GradScaler defaults can vary
        scaler = SafeGradScaler(enabled=True)
        assert scaler.max_nan_before_fallback == 3
        assert scaler.nan_counter == 0
        assert not scaler.fallback_triggered
        assert scaler._enabled == (True and torch.cuda.is_available())
        assert not scaler.warning_logged

    def test_init_custom_fallback(self, mock_grad_scaler_base_methods):
        """Test init with custom max_nan_before_fallback."""
        scaler = SafeGradScaler(max_nan_before_fallback=5, enabled=True)
        assert scaler.max_nan_before_fallback == 5
        assert scaler._enabled == (True and torch.cuda.is_available())

    def test_init_disabled(self, mock_grad_scaler_base_methods):
        """Test init when scaler is disabled from the start."""
        scaler = SafeGradScaler(enabled=False)
        assert not scaler._enabled
        assert not scaler.fallback_triggered # Fallback not triggered just by init args

    # --- scale() Tests --- #

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_normal(self, mock_cuda_available, mock_grad_scaler_base_methods):
        """Test scale() with a valid loss when enabled (forcing enabled state)."""
        # Force enabled by patching cuda.is_available
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled # Verify it's enabled due to patch
        loss_tensor = torch.tensor(1.0)
        mock_scaled_loss = mock_grad_scaler_base_methods['scale'].return_value

        scaled_loss = scaler.scale(loss_tensor)

        mock_grad_scaler_base_methods['scale'].assert_called_once_with(scaler, loss_tensor)
        assert scaled_loss == mock_scaled_loss
        assert scaler.nan_counter == 0 # Counter remains 0
        assert not scaler.warning_logged

    def test_scale_disabled(self, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() when the scaler is disabled."""
        scaler = SafeGradScaler(enabled=False)
        loss_tensor = torch.tensor(1.0)

        scaled_loss = scaler.scale(loss_tensor)
        scaled_loss_again = scaler.scale(loss_tensor) # Call again to check warning log count

        mock_grad_scaler_base_methods['scale'].assert_not_called()
        assert scaled_loss is loss_tensor # Returns unscaled loss
        assert scaled_loss_again is loss_tensor
        mock_logger.warning.assert_called_once_with("Mixed precision is disabled. Loss scaling skipped.")
        assert scaler.warning_logged

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_nan_loss_increment(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() increments counter on NaN and returns unscaled loss."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True, max_nan_before_fallback=3)
        assert scaler._enabled
        loss_tensor_nan = torch.tensor(float('nan'))

        scaled_loss = scaler.scale(loss_tensor_nan)

        mock_grad_scaler_base_methods['scale'].assert_not_called()
        assert scaled_loss is loss_tensor_nan # Returns unscaled loss
        assert scaler.nan_counter == 1
        mock_logger.warning.assert_called_once()
        assert "NaN/Inf detected" in mock_logger.warning.call_args[0][0]
        assert "occurrence 1/3" in mock_logger.warning.call_args[0][0]
        assert not scaler.fallback_triggered
        assert scaler._enabled

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_inf_loss_increment(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() increments counter on Inf and returns unscaled loss."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True, max_nan_before_fallback=3)
        assert scaler._enabled
        loss_tensor_inf = torch.tensor(float('inf'))

        scaled_loss = scaler.scale(loss_tensor_inf)

        mock_grad_scaler_base_methods['scale'].assert_not_called()
        assert scaled_loss is loss_tensor_inf
        assert scaler.nan_counter == 1
        mock_logger.warning.assert_called_once()
        assert "NaN/Inf detected" in mock_logger.warning.call_args[0][0]
        assert "occurrence 1/3" in mock_logger.warning.call_args[0][0]
        assert not scaler.fallback_triggered
        assert scaler._enabled

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_nan_counter_decrement(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() decrements counter when a valid loss follows NaN/Inf."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        loss_tensor_nan = torch.tensor(float('nan'))
        loss_tensor_valid = torch.tensor(1.0)
        mock_scaled_loss = mock_grad_scaler_base_methods['scale'].return_value

        scaler.scale(loss_tensor_nan) # nan_counter = 1
        assert scaler.nan_counter == 1
        mock_logger.warning.assert_called_once()
        mock_grad_scaler_base_methods['scale'].reset_mock()

        # Scale with valid loss
        scaled_loss = scaler.scale(loss_tensor_valid)
        assert scaler.nan_counter == 0 # Counter decremented
        assert scaled_loss == mock_scaled_loss
        mock_grad_scaler_base_methods['scale'].assert_called_once_with(scaler, loss_tensor_valid)
        # No new warning
        mock_logger.warning.assert_called_once()

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_triggers_fallback(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() disables scaler after max NaN/Inf occurrences."""
        max_nan = 2
        # Force enabled
        scaler = SafeGradScaler(enabled=True, max_nan_before_fallback=max_nan)
        assert scaler._enabled
        loss_tensor_nan = torch.tensor(float('nan'))

        # Trigger fallback
        for i in range(max_nan):
            scaled_loss = scaler.scale(loss_tensor_nan)
            assert scaled_loss is loss_tensor_nan
            assert scaler.nan_counter == i + 1
            assert mock_logger.warning.call_count == i + 1
            if i < max_nan - 1:
                 assert not scaler.fallback_triggered
                 assert scaler._enabled
            else:
                 # Last call should trigger fallback
                 assert scaler.fallback_triggered
                 assert not scaler._enabled
                 mock_logger.error.assert_called_once()
                 assert "Disabling mixed precision" in mock_logger.error.call_args[0][0]

        # Subsequent call when disabled
        mock_logger.warning.reset_mock()
        mock_logger.error.reset_mock()
        scaled_loss = scaler.scale(loss_tensor_nan)
        assert scaled_loss is loss_tensor_nan
        assert not scaler._enabled
        assert scaler.fallback_triggered
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()
        assert scaler.nan_counter == max_nan # Counter doesn't change once fallback is triggered

    # --- step() Tests --- #

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_normal(self, mock_cuda_available, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() calls super().step when enabled (forcing enabled state)."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        mock_step = mock_grad_scaler_base_methods['step']
        mock_step.return_value = "Step Result"

        result = scaler.step(mock_optimizer, 1, key='val')

        mock_step.assert_called_once_with(scaler, mock_optimizer, 1, key='val')
        assert result == "Step Result"

    def test_step_disabled(self, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() calls optimizer.step directly when disabled."""
        scaler = SafeGradScaler(enabled=False)
        mock_optimizer.step.return_value = "Opt Step Result"

        result = scaler.step(mock_optimizer, 1, key='val')

        mock_grad_scaler_base_methods['step'].assert_not_called()
        mock_optimizer.step.assert_called_once_with(1, key='val')
        assert result == "Opt Step Result"

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_nan_inf_error(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() handles NaN/Inf RuntimeError from super().step."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        error_message = "Gradient contains Inf or NaN"
        mock_step = mock_grad_scaler_base_methods['step']
        mock_step.side_effect = RuntimeError(error_message)

        result = scaler.step(mock_optimizer)

        mock_step.assert_called_once_with(scaler, mock_optimizer)
        assert not scaler._enabled # Scaler should be disabled
        assert scaler.fallback_triggered
        assert scaler.warning_logged # Should be set by fallback trigger
        mock_logger.error.assert_called_once()
        assert "NaN/Inf detected during optimizer step" in mock_logger.error.call_args[0][0]
        assert error_message in mock_logger.error.call_args[0][0]
        mock_logger.warning.assert_called_once_with("Skipping optimizer step for this iteration due to invalid gradients.")
        mock_optimizer.zero_grad.assert_called_once() # Gradients zeroed
        mock_optimizer.step.assert_not_called() # Optimizer step skipped
        assert result is None

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_other_runtime_error(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() re-raises other RuntimeErrors."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        error_message = "Some other runtime error"
        mock_step = mock_grad_scaler_base_methods['step']
        mock_step.side_effect = RuntimeError(error_message)

        with pytest.raises(RuntimeError, match=error_message):
            scaler.step(mock_optimizer)

        mock_step.assert_called_once_with(scaler, mock_optimizer)
        mock_logger.error.assert_not_called() # Should not log the NaN fallback message
        assert not scaler.fallback_triggered
        assert scaler._enabled # Scaler remains enabled

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_unexpected_error(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() re-raises unexpected non-RuntimeErrors."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        error_message = "Something else went wrong"
        mock_step = mock_grad_scaler_base_methods['step']
        mock_step.side_effect = ValueError(error_message) # Different error type

        with pytest.raises(ValueError, match=error_message):
            scaler.step(mock_optimizer)

        mock_step.assert_called_once_with(scaler, mock_optimizer)
        mock_logger.error.assert_called_once()
        assert "Unexpected error during SafeGradScaler.step" in mock_logger.error.call_args[0][0]
        assert not scaler.fallback_triggered
        assert scaler._enabled

    # --- update() Tests --- #

    @patch('torch.cuda.is_available', return_value=True)
    def test_update_normal(self, mock_cuda_available, mock_grad_scaler_base_methods):
        """Test update() calls super().update when enabled (forcing enabled state)."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        mock_update = mock_grad_scaler_base_methods['update']

        scaler.update(new_scale=128.0)

        mock_update.assert_called_once_with(scaler, new_scale=128.0)

    def test_update_disabled(self, mock_grad_scaler_base_methods):
        """Test update() does nothing when disabled."""
        scaler = SafeGradScaler(enabled=False)
        mock_update = mock_grad_scaler_base_methods['update']

        scaler.update()

        mock_update.assert_not_called()

    # --- state_dict() / load_state_dict() Tests --- #

    def test_state_dict(self, mock_grad_scaler_base_methods):
        """Test state_dict includes custom attributes."""
        scaler = SafeGradScaler(enabled=True)
        scaler.nan_counter = 2
        scaler.fallback_triggered = True
        # Mock the base state dict
        base_state = {'scale': torch.tensor(1024.0), '_step': 5}
        mock_grad_scaler_base_methods['state_dict'].return_value = base_state.copy()

        state = scaler.state_dict()

        mock_grad_scaler_base_methods['state_dict'].assert_called_once_with(scaler)
        assert state['fallback_triggered'] is True
        assert state['nan_counter'] == 2
        assert state['scale'] == base_state['scale'] # Includes base state
        assert state['_step'] == base_state['_step']

    def test_load_state_dict_normal(self, mock_grad_scaler_base_methods):
        """Test loading a state dict without fallback triggered."""
        scaler = SafeGradScaler(enabled=True)
        state = {
            'scale': torch.tensor(512.0),
            '_step': 3,
            'fallback_triggered': False,
            'nan_counter': 1
        }

        scaler.load_state_dict(state.copy()) # Pass copy to avoid modification

        mock_grad_scaler_base_methods['load_state_dict'].assert_called_once_with(scaler, state)
        assert not scaler.fallback_triggered
        assert scaler.nan_counter == 1
        assert scaler._enabled == (True and torch.cuda.is_available())

    def test_load_state_dict_fallback(self, mock_grad_scaler_base_methods):
        """Test loading a state dict with fallback triggered disables scaler."""
        scaler = SafeGradScaler(enabled=True)
        state = {
            'scale': torch.tensor(256.0),
            '_step': 10,
            'fallback_triggered': True,
            'nan_counter': 3
        }

        scaler.load_state_dict(state.copy())

        mock_grad_scaler_base_methods['load_state_dict'].assert_called_once_with(scaler, state)
        assert scaler.fallback_triggered
        assert scaler.nan_counter == 3
        assert not scaler._enabled # Should be disabled after loading
        assert scaler.warning_logged # Should be set when loading fallback state

    def test_load_state_dict_missing_keys(self, mock_grad_scaler_base_methods):
        """Test loading state dict with missing custom keys (uses defaults)."""
        scaler = SafeGradScaler(enabled=True)
        state = {
            'scale': torch.tensor(128.0),
            '_step': 1
            # Missing fallback_triggered and nan_counter
        }

        scaler.load_state_dict(state.copy())

        mock_grad_scaler_base_methods['load_state_dict'].assert_called_once_with(scaler, state)
        assert not scaler.fallback_triggered # Defaults to False
        assert scaler.nan_counter == 0 # Defaults to 0
        assert scaler._enabled == (True and torch.cuda.is_available())

    # ... to be added 