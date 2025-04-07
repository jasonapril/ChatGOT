# src/craft/ Package

This directory contains the core source code for the Craft framework.

## Sub-packages:

- `config/`: Pydantic schemas for configuration validation.
- `models/`: Base classes and implementations for neural network models.
- `data/`: Modules for data loading, processing, and tokenization.
- `training/`: Core logic for model training, evaluation, checkpointing, and callbacks.
- `cli/`: Implementation of the command-line interface using Typer.
- `utils/`: Common utility functions used across the framework.

## Purpose

This directory contains the core Python source code for the `craft` library/package. It includes all modules related to model definitions, data handling, training logic, configuration management, utilities, and the command-line interface.

## Structure

The code is organized into modules based on functionality:
-   `craft/`
    -   `__init__.py`: Makes `craft` a package, defines the public API (`__all__`, version, etc.).
    -   `cli/`: Code related to the command-line interface (using Typer).
    -   `config/`: Configuration validation using Pydantic schemas (`schemas.py`) that interface with the root `conf/` YAML files.
    -   `data/`: Data loading (datasets, dataloaders), processing, and tokenization.
    -   `models/`: Model architecture implementations (e.g., Transformer) and related components (layers, embeddings).
    -   `training/`: Core training logic (`Trainer` class, `TrainingLoop` class), callbacks (`callbacks.py`), checkpointing (`checkpointing.py`), evaluation, and generation utilities. Optimizers and schedulers are configured via Hydra and handled by the Trainer.
    -   `utils/`: Common utility functions (logging, seeding, device handling, file I/O, etc.) used across the project.
    -   `README.md`: This file.

    Future considerations:
    -   `performance/`: Performance monitoring (throughput, memory), instrumentation, and visualization utilities.

## Guidelines

- Follow PEP 8 style guidelines.
- Write clear docstrings for all public modules, classes, and functions.
- Ensure modules have clear responsibilities and interfaces.
- Add unit and integration tests for this code in the top-level `tests/` directory.
- Imports should generally use absolute paths from `craft` (e.g., `from craft.models import ...`). 