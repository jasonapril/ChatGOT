"""Configs module for the project.

This module contains configuration files for the project.

Organization:
- models/: Model architecture configs
- training/: Training hyperparameter configs
- data/: Data processing configs
- pipeline/: Pipeline configs
- experiments/: Experiment configs
"""

from pathlib import Path

CONFIG_PATH = Path(__file__).parent.absolute()
__all__ = ["CONFIG_PATH"]

"""Configuration module for the project.""" 