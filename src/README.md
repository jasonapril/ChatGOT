# Source Code (`src/`)

This directory contains the core Python source code for the Craft project.

## Key Subdirectories:

*   `cli/`: Command-line interface entry points (e.g., `run.py` using Typer/Hydra).
*   `data/`: Modules related to dataset loading, processing, and management (e.g., `CharDataset`, data builders).
*   `models/`: Model definitions (e.g., `base.py`, `gpt_decoder.py`, `factory.py`).
*   `performance/`: Code related to performance monitoring, instrumentation, and potentially visualization during runs (e.g., `throughput_core.py`).
*   `training/`: Core logic for the training loop, optimizers, schedulers, callbacks, and generation algorithms (e.g., `trainer.py`, `optimizations.py`, `generation.py`).
*   `utils/`: General utility functions supporting various parts of the codebase (e.g., logging, I/O, checkpoints, metrics). 