# Craft Package (`src/craft/`)

## Purpose

This directory contains the core Python source code for the `craft` library/package. It includes all modules related to model definitions, data handling, training logic, configuration management, utilities, and the command-line interface.

## Structure

The code is organized into modules based on functionality:
-   `craft/`
    -   `__init__.py`: Makes `craft` a package, defines the public API (`__all__`, version, etc.).
    -   `cli/`: Code related to the command-line interface (using Click or Argparse).
    -   `config/`: Configuration loading, validation, and management utilities (interacts with root `conf/` files).
    -   `data/`: Data loading (datasets, dataloaders), processing, and vocabulary management.
    -   `models/`: Model architecture implementations (e.g., Transformer) and related components (layers, embeddings).
    -   `performance/`: Performance monitoring (throughput, memory), instrumentation, and visualization utilities.
    -   `training/`: Training loops (`Trainer` class), evaluation logic, optimizers, schedulers, loss functions.
    -   `utils/`: Common utility functions (logging, seeding, device handling, metrics, etc.) used across the project.
    -   `README.md`: This file.

## Guidelines

- Follow PEP 8 style guidelines.
- Write clear docstrings for all public modules, classes, and functions.
- Ensure modules have clear responsibilities and interfaces.
- Add unit and integration tests for this code in the top-level `tests/` directory.
- Imports should generally use absolute paths from `craft` (e.g., `from craft.models import ...`). 