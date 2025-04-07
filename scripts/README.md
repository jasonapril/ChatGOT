# Scripts (`scripts/`)

## Purpose

This directory contains standalone Python scripts, primarily for auxiliary development tasks, debugging utilities, or potentially legacy workflows.

**Note:** The primary interface for core operations like model training, text generation, and data preparation is now the unified Command-Line Interface (CLI). Use `python -m craft.cli.run --help` for details on available CLI commands.

Scripts in this folder may include:

-   Utilities for inspecting model parameters or vocabulary (`check_model_params.py`, `check_vocab.py`).
-   Helpers for specific development or testing scenarios (`timeout_test_runner.py`).
-   Scripts for managing checkpoints (`get_latest_checkpoint.py`).
-   Potentially legacy scripts (`train.py`, `generate.py`, `train_tokenizer.py`) whose functionality has largely been integrated into the CLI.

## Usage

Consult the individual script's command-line help (usually via `python scripts/<script_name>.py --help`) for specific usage instructions.

## Guidelines

-   New core functionality should generally be added as a command to the main CLI (`src/craft/cli/`) rather than as a new script here.
-   Existing scripts should clearly document their purpose and usage.
-   Scripts can leverage functions and classes defined in `src/craft/`.