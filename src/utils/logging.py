"""
Logging utilities for the project.
"""
import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Optional


def setup_logger(
    name: str = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console output.
    
    Args:
        name: Logger name (None for root logger)
        level: Logging level
        log_file: Path to log file (None for no file output)
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to parent loggers
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if log_file provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_section_header(logger: logging.Logger, title: str, width: int = 80) -> None:
    """
    Log a section header with decorative formatting.
    
    Args:
        logger: Logger to use
        title: Title text
        width: Width of the header
    """
    separator = "=" * width
    padding = " " * ((width - len(title) - 2) // 2)
    header = f"{separator}\n{padding}{title}{padding}\n{separator}"
    logger.info(f"\n{header}")


def force_flush_logs() -> None:
    """Force flush all handlers for all loggers."""
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.flush()


def format_time(seconds: float) -> str:
    """
    Format seconds as a human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "2h 30m 15s")
    """
    if seconds < 0:
        return "0s"
    
    delta = timedelta(seconds=int(seconds))
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts) 