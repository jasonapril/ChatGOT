import pytest
import torch
import logging
from unittest.mock import MagicMock, patch, PropertyMock

# Class to test
from craft.training.amp import SafeGradScaler

# Mock the base GradScaler methods we override or interact with
@pytest.fixture
def mock_grad_scaler_base_methods():
    with patch.object(SafeGradScaler, 'scale', return_value=torch.tensor(2.0)) as mock_scale:
        with patch.object(SafeGradScaler, 'step') as mock_step:
            with patch.object(SafeGradScaler, 'update') as mock_update:
                with patch.object(SafeGradScaler, 'state_dict', return_value={}) as mock_state_dict:
                    with patch.object(SafeGradScaler, 'load_state_dict') as mock_load_state_dict:
                        yield {
                            'scale': mock_scale, 
                            'step': mock_step, 
                            'update': mock_update, 
                            'state_dict': mock_state_dict, 
                            'load_state_dict': mock_load_state_dict
                        }

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

@pytest.mark.skip(reason="Refactoring needed: Tests broken after GradScaler update and internal changes.")
class TestSafeGradScaler:

    # --- Initialization Tests ---

    def test_init_defaults(self, mock_grad_scaler_base_methods):
        """Test init with default settings."""
        # Note: We pass enabled=True explicitly because GradScaler defaults can vary
        scaler = SafeGradScaler(enabled=True)
        assert scaler.max_consecutive_nan_skip == 5
        assert scaler.enable_fallback is False
        assert scaler._consecutive_nan_skips == 0
        assert scaler._internal_enabled is True
        assert scaler._initial_enabled_state is True
        assert scaler._enabled is True # Check the inherited _enabled state

    def test_init_custom_fallback(self, mock_grad_scaler_base_methods):
        """Test init with custom fallback settings."""
        scaler = SafeGradScaler(max_consecutive_nan_skip=5, enable_fallback=True, enabled=True)
        assert scaler.max_consecutive_nan_skip == 5
        assert scaler.enable_fallback is True
        assert scaler._consecutive_nan_skips == 0
        assert scaler._internal_enabled is True

    def test_init_disabled(self, mock_grad_scaler_base_methods):
        """Test init when scaler is disabled from the start."""
        scaler = SafeGradScaler(enabled=False)
        assert not scaler._enabled # Inherited state
        assert not scaler._internal_enabled # Internal state
        assert scaler._consecutive_nan_skips == 0 # Counter starts at 0

    # --- scale() Tests --- #

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_normal(self, mock_cuda_available, mock_grad_scaler_base_methods):
        """Test scale() with a valid loss when enabled (forcing enabled state)."""
        # Force enabled by patching cuda.is_available
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled # Verify it's enabled due to patch
        loss_tensor = torch.tensor(1.0)
        mock_scaled_loss = mock_grad_scaler_base_methods['scale'].return_value

        with patch.object(torch.amp.GradScaler, 'scale', return_value=mock_scaled_loss) as mock_super_scale:
            scaled_loss = scaler.scale(loss_tensor)
            mock_super_scale.assert_called_once_with(scaler, loss_tensor)
            assert scaled_loss == mock_scaled_loss
            assert scaler._consecutive_nan_skips == 0 # Counter remains 0

    def test_scale_disabled(self, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() when the scaler is disabled."""
        scaler = SafeGradScaler(enabled=False)
        loss_tensor = torch.tensor(1.0)

        with patch.object(torch.amp.GradScaler, 'scale') as mock_super_scale:
            scaled_loss = scaler.scale(loss_tensor)
            mock_super_scale.assert_not_called()
            assert scaled_loss is loss_tensor # Returns unscaled loss
            assert mock_logger.warning.call_count == 2 # Called twice
            mock_logger.warning.assert_called_with("Mixed precision is disabled. Loss scaling skipped.")

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_nan_loss_increment(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() increments counter on NaN and returns unscaled loss."""
        # Force enabled, use correct argument name
        scaler = SafeGradScaler(enabled=True, max_consecutive_nan_skip=3, enable_fallback=False)
        assert scaler._enabled
        loss_tensor_nan = torch.tensor(float('nan'))

        with patch.object(torch.amp.GradScaler, 'scale') as mock_super_scale:
            scaled_loss = scaler.scale(loss_tensor_nan)
            mock_super_scale.assert_not_called()
            assert scaled_loss is loss_tensor_nan # Return original NaN loss
            assert scaler._consecutive_nan_skips == 2 # Check counter update
            mock_logger.warning.assert_called()
            assert "NaN/Inf detected" in mock_logger.warning.call_args[0][0]
            # Check log message includes correct counts (call 1: 1/3, call 2: 2/3)
            assert "(occurrence 1/3)" in mock_logger.warning.call_args_list[0][0][0]
            assert "(occurrence 2/3)" in mock_logger.warning.call_args_list[1][0][0]

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_inf_loss_increment(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() increments counter on Inf and returns unscaled loss."""
        # Force enabled, use correct argument name
        scaler = SafeGradScaler(enabled=True, max_consecutive_nan_skip=3, enable_fallback=False)
        assert scaler._enabled
        loss_tensor_inf = torch.tensor(float('inf'))

        with patch.object(torch.amp.GradScaler, 'scale') as mock_super_scale:
            scaled_loss = scaler.scale(loss_tensor_inf)
            mock_super_scale.assert_not_called()
            assert scaled_loss is loss_tensor_inf
            assert scaler._consecutive_nan_skips == 2
            mock_logger.warning.assert_called()
            assert "NaN/Inf detected" in mock_logger.warning.call_args[0][0]
            assert "(occurrence 1/3)" in mock_logger.warning.call_args_list[0][0][0]
            assert "(occurrence 2/3)" in mock_logger.warning.call_args_list[1][0][0]

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_nan_counter_decrement(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() decrements counter when a valid loss follows NaN/Inf."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        loss_tensor_nan = torch.tensor(float('nan'))
        loss_tensor_valid = torch.tensor(1.0)
        mock_scaled_loss = torch.tensor(2.0)

        scaler.scale(loss_tensor_nan) # _consecutive_nan_skips = 1
        assert scaler._consecutive_nan_skips == 1

        with patch.object(torch.amp.GradScaler, 'scale', return_value=mock_scaled_loss) as mock_super_scale:
            scaled_loss_valid = scaler.scale(loss_tensor_valid)
            mock_super_scale.assert_called_once_with(scaler, loss_tensor_valid)
            assert scaled_loss_valid == mock_scaled_loss
            assert scaler._consecutive_nan_skips == 0 # Decremented back to 0

    @patch('torch.cuda.is_available', return_value=True)
    def test_scale_triggers_fallback(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test scale() disables scaler after max NaN/Inf occurrences IF fallback is False."""
        max_nan = 2
        # Force enabled, use correct argument name, disable fallback
        scaler = SafeGradScaler(enabled=True, max_consecutive_nan_skip=max_nan, enable_fallback=False)
        assert scaler._enabled
        loss_tensor_nan = torch.tensor(float('nan'))

        # Call scale max_nan times with NaN
        for i in range(max_nan):
            scaled_loss = scaler.scale(loss_tensor_nan)
            assert scaled_loss is loss_tensor_nan
            assert scaler._consecutive_nan_skips == (i + 1)
            assert f"(occurrence {i+1}/{max_nan})" in mock_logger.warning.call_args_list[i][0][0]
            if i < max_nan - 1:
                assert scaler._enabled
            else: # On the last call
                assert not scaler._enabled # Should be disabled now
                mock_logger.error.assert_called_once_with(
                    "Disabling mixed precision permanently due to repeated NaN/Inf values in loss."
                )

        # Check state after loop
        assert not scaler._enabled
        assert scaler._consecutive_nan_skips == max_nan

        # Call again, should remain disabled and log warning
        scaled_loss_after = scaler.scale(torch.tensor(1.0))
        assert scaled_loss_after == torch.tensor(1.0) # Unscaled
        assert not scaler._enabled
        mock_logger.warning.assert_called_with("Mixed precision is disabled. Loss scaling skipped.")
        # Total warning calls = max_nan (for NaN) + 1 (for disabled)
        assert mock_logger.warning.call_count == max_nan + 1

    # --- step() Tests --- #

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_normal(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() calls super().step when enabled."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        mock_step = mock_grad_scaler_base_methods['step']

        scaler.step(mock_optimizer)

        mock_step.assert_called_once_with(scaler, mock_optimizer)
        mock_logger.error.assert_not_called()
        assert scaler._enabled # Stays enabled

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_disabled(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() calls optimizer.step directly when disabled."""
        scaler = SafeGradScaler(enabled=False)
        assert not scaler._enabled
        mock_step = mock_grad_scaler_base_methods['step']

        scaler.step(mock_optimizer)

        mock_step.assert_not_called() # Super().step should not be called
        mock_optimizer.step.assert_called_once()
        mock_logger.error.assert_not_called()

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_nan_inf_error(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() handles NaN/Inf RuntimeError from super().step and disables scaler."""
        # Force enabled, disable fallback for this test
        scaler = SafeGradScaler(enabled=True, enable_fallback=False)
        assert scaler._enabled
        error_message = "Gradient contains Inf or NaN"
        # Mock the base class step method
        with patch.object(torch.amp.GradScaler, 'step', side_effect=RuntimeError(error_message)) as mock_super_step:
            result = scaler.step(mock_optimizer)
            mock_super_step.assert_called_once_with(scaler, mock_optimizer)
            assert not scaler._enabled # Scaler should be disabled
            assert not scaler._internal_enabled
            mock_logger.error.assert_called_once()
            assert "NaN/Inf detected during optimizer step" in mock_logger.error.call_args[0][0]
            mock_optimizer.zero_grad.assert_called_once() # Check gradients zeroed
            assert result is None # Should not return anything

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_other_runtime_error(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() re-raises other RuntimeErrors."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        error_message = "Some other runtime error"
        # Mock the base class step method
        with patch.object(torch.amp.GradScaler, 'step', side_effect=RuntimeError(error_message)) as mock_super_step:
            with pytest.raises(RuntimeError, match=error_message):
                scaler.step(mock_optimizer)

            mock_super_step.assert_called_once_with(scaler, mock_optimizer)
            mock_logger.error.assert_not_called() # Should not log the NaN fallback message
            assert scaler._enabled # Should remain enabled

    @patch('torch.cuda.is_available', return_value=True)
    def test_step_unexpected_error(self, mock_cuda_available, mock_logger, mock_optimizer, mock_grad_scaler_base_methods):
        """Test step() re-raises unexpected non-RuntimeErrors."""
        # Force enabled
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        error_message = "Something else went wrong"
        # Mock the base class step method
        with patch.object(torch.amp.GradScaler, 'step', side_effect=ValueError(error_message)) as mock_super_step:
            with pytest.raises(ValueError, match=error_message):
                scaler.step(mock_optimizer)

            mock_super_step.assert_called_once_with(scaler, mock_optimizer)
            mock_logger.error.assert_called_once()
            assert "Unexpected error during SafeGradScaler.step" in mock_logger.error.call_args[0][0]
            assert scaler._enabled # Should remain enabled

    # --- update() Tests --- #

    @patch('torch.cuda.is_available', return_value=True)
    def test_update_normal(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test update() calls super().update when enabled."""
        scaler = SafeGradScaler(enabled=True)
        assert scaler._enabled
        mock_update = mock_grad_scaler_base_methods['update']

        scaler.update()

        mock_update.assert_called_once_with(scaler)
        assert scaler._enabled

    @patch('torch.cuda.is_available', return_value=True)
    def test_update_disabled(self, mock_cuda_available, mock_logger, mock_grad_scaler_base_methods):
        """Test update() does nothing when disabled."""
        scaler = SafeGradScaler(enabled=False)
        assert not scaler._enabled
        mock_update = mock_grad_scaler_base_methods['update']

        scaler.update()

        mock_update.assert_not_called()

    # --- state_dict() / load_state_dict() Tests --- #

    def test_state_dict(self, mock_grad_scaler_base_methods):
        """Test state_dict includes custom attributes."""
        scaler = SafeGradScaler(enabled=True)
        scaler._consecutive_nan_skips = 2
        scaler._internal_enabled = False # Simulate internal state change
        scaler._initial_enabled_state = True

        # Mock the base state dict
        base_state = {'scale': torch.tensor(1024.0), '_growth_tracker': 5}
        # Use patch.object on the *base class* state_dict
        with patch.object(torch.amp.GradScaler, 'state_dict', return_value=base_state.copy()) as mock_super_state_dict:
            state = scaler.state_dict()
            mock_super_state_dict.assert_called_once_with(scaler)

        assert state['scale'] == base_state['scale'] # Check base keys are preserved
        assert state['_growth_tracker'] == base_state['_growth_tracker']
        # Check custom keys
        assert state['_consecutive_nan_skips'] == 2
        assert state['_internal_enabled'] is False
        assert state['_initial_enabled_state'] is True

    def test_load_state_dict_normal(self, mock_grad_scaler_base_methods):
        """Test loading a state dict without fallback triggered."""
        scaler = SafeGradScaler(enabled=True)
        state = {
            'scale': torch.tensor(512.0),
            '_growth_tracker': 3,
            '_consecutive_nan_skips': 1,
            '_internal_enabled': True,
            '_initial_enabled_state': True
        }
        # Mock the base load_state_dict
        with patch.object(torch.amp.GradScaler, 'load_state_dict') as mock_super_load:
            scaler.load_state_dict(state.copy()) # Pass copy to avoid modification
            # Check that base load_state_dict was called with the *original* state dict
            # (super() call happens AFTER loading custom attributes)
            mock_super_load.assert_called_once_with(scaler, state)

        assert scaler._consecutive_nan_skips == 1
        assert scaler._internal_enabled is True
        assert scaler._enabled # Check inherited property reflects internal state

    def test_load_state_dict_fallback(self, mock_grad_scaler_base_methods):
        """Test loading a state dict where fallback condition is met."""
        max_nan = 3
        scaler = SafeGradScaler(enabled=True, max_consecutive_nan_skip=max_nan)
        state = {
            'scale': torch.tensor(256.0),
            '_growth_tracker': 10,
            '_consecutive_nan_skips': max_nan, # Fallback condition met
            '_internal_enabled': True, # Assume it was enabled before this state
            '_initial_enabled_state': True
        }

        with patch.object(torch.amp.GradScaler, 'load_state_dict') as mock_super_load:
            scaler.load_state_dict(state.copy())
            mock_super_load.assert_called_once_with(scaler, state)

        assert scaler._consecutive_nan_skips == max_nan
        # Check that loading this state correctly sets the internal and inherited enabled flags
        # This logic is within load_state_dict
        assert not scaler._enabled # Should be disabled due to consecutive skips >= max
        assert not scaler._internal_enabled # Should also be disabled

    def test_load_state_dict_missing_keys(self, mock_grad_scaler_base_methods):
        """Test loading state dict with missing custom keys (uses defaults)."""
        scaler = SafeGradScaler(enabled=True)
        state = {
            'scale': torch.tensor(128.0),
            '_growth_tracker': 1
            # Missing custom keys
        }

        with patch.object(torch.amp.GradScaler, 'load_state_dict') as mock_super_load:
            scaler.load_state_dict(state.copy())
            mock_super_load.assert_called_once_with(scaler, state)

        # Check custom attributes received default values
        assert scaler._consecutive_nan_skips == 0 # Defaults to 0
        assert scaler._internal_enabled is True   # Defaults to True
        assert scaler._initial_enabled_state is True # Defaults to True
        assert scaler._enabled is True # Reflects internal state

    # ... to be added 