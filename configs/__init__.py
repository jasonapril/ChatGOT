"""
Configs module for ChatGoT.

This module contains configuration files for the ChatGoT project.
All configurations are managed through Hydra.
"""

from pathlib import Path

CONFIG_PATH = Path(__file__).parent.absolute()
__all__ = ["CONFIG_PATH"]

"""Configuration module for ChatGoT.""" 