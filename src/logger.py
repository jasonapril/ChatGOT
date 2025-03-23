"""
Logger Module
============

This module provides an enhanced logging system for the training process. It focuses on:

1. Robust, platform-independent logging configurations
2. Immediate log flushing for real-time monitoring
3. Both console and file logging support
4. Emergency error handling to ensure logs are always written

The module was designed to provide instant visibility during long-running training 
processes without relying on external monitoring tools. Log entries are designed
to be immediately flushed to disk, which is crucial for monitoring training progress
and detecting issues early.

Design Principles:
- Immediate visibility: All logs are flushed immediately to enable real-time monitoring
- Error resilience: Multiple fallback mechanisms for writing logs
- Compatibility: Works across different operating systems, including Windows
- Minimal dependencies: Only uses standard library components
"""

import os
import sys
import logging
import datetime
import io

def setup_logger(log_file=None, level=logging.INFO):
    """
    Setup logging configuration for both console and file outputs.
    Ensures immediate visibility of log messages with failsafe mechanisms.
    
    Args:
        log_file (str, optional): Path to log file. If None, only console logging is used.
        level (int, optional): Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured root logger
    """
    # Create a formatter that includes timestamps
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove all handlers to avoid duplicates (if running multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with immediate flushing
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Ensure console output is immediately visible
    console_handler.stream = sys.stdout  # Use stdout explicitly
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        try:
            # Use a special file handler that flushes immediately
            file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='backslashreplace', mode='a')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            
            # Hack to ensure Windows flushes on every log
            orig_emit = file_handler.emit
            def new_emit(record):
                orig_emit(record)
                file_handler.flush()
                try:
                    # For Windows, force the OS to write to disk
                    if hasattr(file_handler.stream, 'fileno'):
                        os.fsync(file_handler.stream.fileno())
                except (AttributeError, OSError):
                    pass
            file_handler.emit = new_emit
            
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file: {os.path.abspath(log_file)}")
            
            # Write an immediate test message and force flush
            logging.info(f"=== LOGGING STARTED AT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            force_flush_logs()
        except Exception as e:
            logging.error(f"Failed to set up file logging: {str(e)}")
            print(f"Error setting up log file: {str(e)}", flush=True)
    
    # Log startup info with timestamp
    logging.info(f"Logger initialized with {'console and file' if log_file else 'console only'} output")
    
    # Force Python's stdout/stderr to be line-buffered for more immediate feedback
    sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
    sys.stderr = io.TextIOWrapper(open(sys.stderr.fileno(), 'wb', 0), write_through=True)
    
    return root_logger

def force_flush_logs():
    """
    Force flush all logging handlers and stdout.
    
    This is a critical function for real-time log monitoring, ensuring that
    logs are written to disk immediately. This is especially important when:
    1. Training is running for a long time
    2. You're monitoring logs remotely
    3. You need to detect errors in real-time
    
    The function handles potential errors during flushing to maintain robustness.
    """
    # Flush all logging handlers
    for handler in logging.getLogger().handlers:
        try:
            if hasattr(handler, 'flush') and callable(handler.flush):
                handler.flush()
        except Exception:
            pass
    
    # Also flush stdout
    sys.stdout.flush()
    
    # Close and reopen file handlers if applicable (more aggressive flushing)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                # This forces the OS to write to disk
                handler.close()
                handler.stream = open(handler.baseFilename, handler.mode)
            except Exception:
                pass

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string (e.g., "2h 30m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"
    
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def log_section_header(title, width=80):
    """
    Log a section header to make log files more readable.
    
    Args:
        title (str): Section title
        width (int, optional): Width of the header (default: 80)
    """
    logging.info("=" * width)
    logging.info(title.center(width))
    logging.info("=" * width)
    force_flush_logs() 