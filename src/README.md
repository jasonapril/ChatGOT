# Source Code (`src/`)

This directory contains the core Python source code for the project.

## Structure

*   `cli/`: Command-line interface entry points (using Typer/Click).
*   `config/`: Configuration loading and management (using OmegaConf/Hydra helpers).
*   `data/`: Modules related to dataset loading, processing, and management (e.g., `PickledDataset`, `base.py`, `processors.py`).
*   `models/`: Model definitions (e.g., `TransformerModel`) and related utilities.
*   `training/`: Training loop logic (`Trainer`), callbacks, and related utilities.
*   `utils/`: General utility functions (logging, I/O, etc.).
*   `__init__.py`: Package initializer.

Consult `README.md` files within subdirectories for more details. 