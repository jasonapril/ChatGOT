# Scripts (`scripts/`)

## Purpose

This directory contains standalone scripts for various operational tasks related to the project, such as data preparation, running training jobs, evaluation, and potentially deployment or utility tasks. It often serves as the entry point for command-line operations.

## Structure

-   `scripts/`
    -   `train.py`: Script to initiate model training.
    -   `generate.py`: Script to generate samples from a trained model.
    -   `evaluate.py`: Script to run model evaluation.
    -   `prepare_data.py`: Example script for data processing.

## Guidelines

- Scripts should be executable and ideally accept command-line arguments for flexibility.
- Leverage functions and classes defined in `src/` where appropriate.
- Include argument parsing (`argparse`) and clear usage instructions (e.g., via `--help` or comments).