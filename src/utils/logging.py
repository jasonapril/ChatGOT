"""
Logging utilities for Craft.

This module provides utilities for setting up and using loggers throughout the project.
"""
import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> None:
    """
    Configure the root logger with consistent formatting.
    
    Args:
        level: Logging level (e.g., "INFO", "DEBUG")
        log_file: Path to log file (None for no file output)
        console: Whether to log to console
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    setup_logger(
        name=None,  # Root logger
        level=numeric_level,
        log_file=log_file,
        console=console
    )
    
    # Adjust third-party loggers to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level {level}")


def setup_logger(
    name: str = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console output.
    If name is None (root logger), adds handlers instead of clearing existing ones.
    
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
    
    # If configuring a specific logger (not root), clear existing handlers and prevent propagation
    if name is not None:
        logger.propagate = False  
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    # For root logger (name=None), we assume Hydra or another process might have set
    # up handlers, so we ADD handlers instead of clearing.
    # logger.propagate is typically True for root, so leave it.

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check if a similar handler already exists before adding
    has_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in logger.handlers) if log_file else False
    has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers)
    
    # Add file handler if log_file provided and no similar handler exists
    if log_file and not has_file_handler:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        if name is None: logger.info(f"setup_logger added FileHandler for root: {log_file}") # Debug log
    
    # Add console handler if requested and no similar handler exists
    if console and not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        if name is None: logger.info("setup_logger added StreamHandler for root") # Debug log
    
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