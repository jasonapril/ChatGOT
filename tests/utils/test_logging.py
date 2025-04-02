"""""Tests for logging utilities."""""
import pytest
from craft.utils.logging import format_time, log_section_header, force_flush_logs, setup_logger, setup_logging
import logging
from unittest.mock import MagicMock, patch
import os
import sys

# Tests for format_time
@pytest.mark.parametrize(
    "seconds, expected_output",
    [
        (0, "0s"),
        (-10, "0s"),  # Negative input should return 0s
        (15, "15s"),
        (60, "1m 0s"),
        (90, "1m 30s"),
        (3600, "1h 0m 0s"),
        (3665, "1h 1m 5s"),
        (86400, "1d 0h 0m 0s"), # 1 day
        (90061, "1d 1h 1m 1s"), # 1 day, 1 hour, 1 minute, 1 second
        (172800, "2d 0h 0m 0s"), # 2 days
        (0.5, "0s"), # Test rounding down
        (59.9, "59s"), # Test rounding down near minute
    ],
)
def test_format_time(seconds, expected_output):
    """Test that format_time correctly formats seconds into human-readable strings."""
    assert format_time(seconds) == expected_output 

# Tests for log_section_header
def test_log_section_header():
    """Test that log_section_header formats and logs the title correctly."""
    mock_logger = MagicMock()
    title = "Test Title"
    width = 50
    
    log_section_header(mock_logger, title, width)
    
    expected_separator = "=" * width
    padding_len = (width - len(title) - 2) // 2
    padding = " " * padding_len
    expected_title_line = f"{padding}{title}{padding}"
    header_block = f"{expected_separator}\n{expected_title_line}\n{expected_separator}"
    expected_output = f"\n{header_block}"
    
    mock_logger.info.assert_called_once_with(expected_output)

def test_log_section_header_default_width():
    """Test log_section_header with the default width."""
    mock_logger = MagicMock()
    title = "Another Test"
    width = 80 # Default width
    
    log_section_header(mock_logger, title) # Use default width
    
    expected_separator = "=" * width
    padding_len = (width - len(title) - 2) // 2
    padding = " " * padding_len
    expected_title_line = f"{padding}{title}{padding}"
    header_block = f"{expected_separator}\n{expected_title_line}\n{expected_separator}"
    expected_output = f"\n{header_block}"
    
    mock_logger.info.assert_called_once_with(expected_output)

# Tests for force_flush_logs
def test_force_flush_logs():
    """Test that force_flush_logs calls flush on all handlers of registered loggers."""
    # Create mock handlers
    mock_handler1 = MagicMock(spec=logging.Handler)
    mock_handler2 = MagicMock(spec=logging.Handler)
    
    # Create mock loggers and add handlers
    mock_logger1 = MagicMock(spec=logging.Logger)
    mock_logger1.handlers = [mock_handler1]
    
    mock_logger2 = MagicMock(spec=logging.Logger)
    mock_logger2.handlers = [mock_handler2]
    
    # Mock the logger dictionary managed by the logging module
    # Note: We patch 'logging.root.manager.loggerDict' directly if possible,
    # otherwise we might need to patch logging.getLogger if loggerDict isn't directly accessible/mockable.
    # For simplicity here, we assume patching loggerDict works.
    mock_logger_dict = {
        'logger1': mock_logger1,
        'logger2': mock_logger2
    }
    
    # Use patch.dict to temporarily modify the loggerDict during the test
    with patch.dict(logging.root.manager.loggerDict, mock_logger_dict, clear=True):
        force_flush_logs()
        
    # Assert that flush was called on each handler
    mock_handler1.flush.assert_called_once()
    mock_handler2.flush.assert_called_once() 

# Tests for setup_logger
@patch("logging.getLogger")
def test_setup_logger_basic(mock_get_logger):
    """Test basic logger retrieval and level setting."""
    mock_logger = MagicMock()
    mock_logger.handlers = []
    mock_get_logger.return_value = mock_logger
    
    logger_name = "test_logger"
    level = logging.DEBUG
    
    logger = setup_logger(name=logger_name, level=level, console=False) # No handlers
    
    mock_get_logger.assert_called_once_with(logger_name)
    mock_logger.setLevel.assert_called_once_with(level)
    assert logger == mock_logger
    assert not mock_logger.addHandler.called # No handlers added
    assert getattr(logger, 'propagate', True) is False # Propagation disabled for named loggers

@patch("craft.utils.logging.logging.FileHandler")
@patch("logging.getLogger")
@patch("os.makedirs") # Mock os.makedirs to avoid actual directory creation
def test_setup_logger_file_handler(mock_makedirs, mock_get_logger, mock_file_handler, tmp_path):
    """Test setting up a logger with only a file handler."""
    mock_logger = MagicMock()
    mock_logger.handlers = [] # Start with no handlers
    mock_get_logger.return_value = mock_logger
    
    mock_handler_instance = MagicMock()
    mock_file_handler.return_value = mock_handler_instance
    
    logger_name = "file_test_logger"
    level = logging.INFO
    log_file = tmp_path / "test.log"
    log_file_str = str(log_file)
    log_dir = str(log_file.parent)

    logger = setup_logger(name=logger_name, level=level, log_file=log_file_str, console=False)
    
    mock_get_logger.assert_called_once_with(logger_name)
    mock_logger.setLevel.assert_called_once_with(level)
    mock_makedirs.assert_called_once_with(log_dir, exist_ok=True)
    mock_file_handler.assert_called_once_with(log_file_str)
    mock_handler_instance.setFormatter.assert_called_once()
    mock_logger.addHandler.assert_called_once_with(mock_handler_instance)
    assert logger == mock_logger

@patch("logging.StreamHandler")
@patch("logging.getLogger")
def test_setup_logger_console_handler(mock_get_logger, mock_stream_handler):
    """Test setting up a logger with only a console handler."""
    mock_logger = MagicMock()
    mock_logger.handlers = [] # Start with no handlers
    mock_get_logger.return_value = mock_logger
    
    mock_handler_instance = MagicMock()
    mock_stream_handler.return_value = mock_handler_instance
    
    logger_name = "console_test_logger"
    level = logging.WARNING

    logger = setup_logger(name=logger_name, level=level, log_file=None, console=True)
    
    mock_get_logger.assert_called_once_with(logger_name)
    mock_logger.setLevel.assert_called_once_with(level)
    # Check that StreamHandler was called with sys.stdout
    # We need to access the arguments StreamHandler was called with
    #mock_stream_handler.assert_called_once_with(sys.stdout)
    # Check call args more carefully if needed, e.g.:
    found_stdout_handler = False
    for call_args in mock_stream_handler.call_args_list:
        args, kwargs = call_args
        if args and args[0] == sys.stdout:
            found_stdout_handler = True
            break
    assert found_stdout_handler, "StreamHandler was not initialized with sys.stdout"
    
    mock_handler_instance.setFormatter.assert_called_once()
    mock_logger.addHandler.assert_called_once_with(mock_handler_instance)
    assert logger == mock_logger 

@patch("logging.FileHandler")
@patch("logging.StreamHandler")
@patch("logging.getLogger")
@patch("os.makedirs")
def test_setup_logger_both_handlers(mock_makedirs, mock_get_logger, mock_stream_handler, mock_file_handler, tmp_path):
    """Test setting up a logger with both file and console handlers."""
    mock_logger = MagicMock()
    mock_logger.handlers = [] # Start with no handlers
    mock_get_logger.return_value = mock_logger
    
    mock_file_handler_inst = MagicMock()
    mock_file_handler.return_value = mock_file_handler_inst
    mock_stream_handler_inst = MagicMock()
    mock_stream_handler.return_value = mock_stream_handler_inst
    
    logger_name = "both_test_logger"
    level = logging.DEBUG
    log_file = tmp_path / "both.log"
    log_file_str = str(log_file)
    log_dir = os.path.dirname(log_file_str)

    logger = setup_logger(name=logger_name, level=level, log_file=log_file_str, console=True)
    
    mock_get_logger.assert_called_once_with(logger_name)
    mock_logger.setLevel.assert_called_once_with(level)
    
    # File handler checks
    mock_makedirs.assert_called_once_with(log_dir, exist_ok=True)
    mock_file_handler.assert_called_once_with(log_file_str)
    mock_file_handler_inst.setFormatter.assert_called_once()
    
    # Console handler checks
    #mock_stream_handler.assert_called_once_with(sys.stdout)
    found_stdout_handler_call = False
    for call_args in mock_stream_handler.call_args_list:
        args, kwargs = call_args
        if args and args[0] == sys.stdout:
            found_stdout_handler_call = True
            break
    assert found_stdout_handler_call, "StreamHandler not called with sys.stdout"
    mock_stream_handler_inst.setFormatter.assert_called_once()
    
    # Check both handlers were added
    assert mock_logger.addHandler.call_count == 2
    mock_logger.addHandler.assert_any_call(mock_file_handler_inst)
    mock_logger.addHandler.assert_any_call(mock_stream_handler_inst)
    assert logger == mock_logger

@patch("os.makedirs")
@patch("logging.getLogger")
def test_setup_logger_root_logger(mock_get_logger, mock_makedirs, tmp_path):
    """Test setting up the root logger (name=None). Propagation should remain True."""
    # Use real handlers to check if they are added correctly, but mock getLogger
    existing_handler = MagicMock(spec=logging.Handler)
    mock_root_logger = MagicMock(spec=logging.RootLogger)
    mock_root_logger.handlers = [existing_handler]
    mock_root_logger.propagate = True 
    mock_get_logger.return_value = mock_root_logger
    
    level = logging.INFO
    log_file = tmp_path / "root.log"
    log_file_str = str(log_file)

    # Setup root logger (name=None)
    logger = setup_logger(name=None, level=level, log_file=log_file_str, console=True)
    
    mock_get_logger.assert_called_once_with(None) # Called for root logger
    mock_root_logger.setLevel.assert_called_once_with(level)
    
    # Check handlers were ADDED by checking the call args of addHandler
    assert mock_root_logger.addHandler.call_count == 2
    added_handler_types = [type(call_args[0][0]) for call_args in mock_root_logger.addHandler.call_args_list]
    assert logging.FileHandler in added_handler_types
    assert logging.StreamHandler in added_handler_types

    # Ensure propagate was not set to False (it should remain True for root)
    assert getattr(mock_root_logger, 'propagate', True) is True
    assert logger == mock_root_logger

@patch("logging.getLogger")
@patch("os.makedirs")
def test_setup_logger_no_duplicate_handlers(mock_makedirs, mock_get_logger, tmp_path):
    """Test that setup_logger doesn't add duplicate handlers."""
    mock_logger = MagicMock() # Use mock logger
    
    log_file = tmp_path / "duplicate.log"
    log_file_str = str(log_file)
    log_dir = str(log_file.parent)
    
    # Create REAL handlers for the duplicate check inside the function
    # Ensure the directory exists first
    os.makedirs(log_dir, exist_ok=True)
    existing_file_handler = logging.FileHandler(log_file_str) 
    existing_console_handler = logging.StreamHandler(sys.stdout) 
    
    # Assign REAL handlers to the mock logger's list
    mock_logger.handlers = [existing_file_handler, existing_console_handler]
    
    mock_get_logger.return_value = mock_logger
    
    logger_name = "duplicate_test"
    level = logging.INFO

    # Call setup_logger with parameters that match existing handlers
    logger = setup_logger(name=logger_name, level=level, log_file=log_file_str, console=True)
    
    mock_get_logger.assert_called_once_with(logger_name)
    mock_logger.setLevel.assert_called_once_with(level)
    
    # Assert that addHandler was NOT called because handlers already exist
    assert mock_logger.addHandler.call_count == 0 
    assert logger == mock_logger
    
    # Clean up the real file handler we created
    existing_file_handler.close()
    if os.path.exists(log_file_str):
        os.remove(log_file_str)
    if os.path.exists(log_dir) and not os.listdir(log_dir): # Remove dir if empty
         os.rmdir(log_dir)

# Tests for setup_logging
@patch("craft.utils.logging.setup_logger") # Patch setup_logger within the logging module
@patch("logging.getLogger")
def test_setup_logging_defaults(mock_get_logger, mock_setup_logger):
    """Test setup_logging with default arguments."""
    mock_urllib3_logger = MagicMock()
    mock_matplotlib_logger = MagicMock()
    # Configure mock_get_logger to return specific mocks for specific names
    def get_logger_side_effect(name):
        if name == "urllib3":
            return mock_urllib3_logger
        elif name == "matplotlib":
            return mock_matplotlib_logger
        # Return a default mock for other calls if necessary, though setup_logger is patched
        return MagicMock() 
    mock_get_logger.side_effect = get_logger_side_effect

    setup_logging() # Use defaults: level="INFO", log_file=None, console=True

    # Check setup_logger was called for the root logger with correct defaults
    mock_setup_logger.assert_called_once_with(
        name=None,
        level=logging.INFO,
        log_file=None,
        console=True
    )
    
    # Check third-party loggers were adjusted
    mock_get_logger.assert_any_call("urllib3")
    mock_urllib3_logger.setLevel.assert_called_once_with(logging.WARNING)
    
    mock_get_logger.assert_any_call("matplotlib")
    mock_matplotlib_logger.setLevel.assert_called_once_with(logging.WARNING)

@patch("craft.utils.logging.setup_logger")
@patch("logging.getLogger")
def test_setup_logging_custom_args(mock_get_logger, mock_setup_logger, tmp_path):
    """Test setup_logging with custom arguments (DEBUG level, log file)."""
    mock_urllib3_logger = MagicMock()
    mock_matplotlib_logger = MagicMock()
    mock_get_logger.side_effect = lambda name: {
        "urllib3": mock_urllib3_logger,
        "matplotlib": mock_matplotlib_logger
    }.get(name, MagicMock())
    
    log_file = tmp_path / "custom.log"
    log_file_str = str(log_file)
    level = "DEBUG"
    console = False

    setup_logging(level=level, log_file=log_file_str, console=console)

    # Check setup_logger call with custom args
    mock_setup_logger.assert_called_once_with(
        name=None,
        level=logging.DEBUG, # Check level conversion
        log_file=log_file_str,
        console=console
    )
    
    # Check third-party loggers were still adjusted
    mock_get_logger.assert_any_call("urllib3")
    mock_urllib3_logger.setLevel.assert_called_once_with(logging.WARNING)
    mock_get_logger.assert_any_call("matplotlib")
    mock_matplotlib_logger.setLevel.assert_called_once_with(logging.WARNING)

@patch("craft.utils.logging.setup_logger")
@patch("logging.getLogger")
def test_setup_logging_invalid_level(mock_get_logger, mock_setup_logger):
    """Test setup_logging falls back to INFO for an invalid level string."""
    mock_urllib3_logger = MagicMock()
    mock_matplotlib_logger = MagicMock()
    mock_get_logger.side_effect = lambda name: {
        "urllib3": mock_urllib3_logger,
        "matplotlib": mock_matplotlib_logger
    }.get(name, MagicMock())
    
    invalid_level = "INVALID_LEVEL_STRING"

    setup_logging(level=invalid_level)

    # Check setup_logger was called with INFO level as fallback
    mock_setup_logger.assert_called_once_with(
        name=None,
        level=logging.INFO, # Should default to INFO
        log_file=None,
        console=True
    ) 