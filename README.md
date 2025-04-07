# Craft

A modular PyTorch framework designed for building, training, and evaluating AI models, leveraging Hydra for configuration and Pydantic for validation.

## Project Goals

- **Modularity & Extensibility**: Provide a clean architecture that allows easy addition and swapping of components like models, datasets, tokenizers, and training routines.
- **Configuration Driven**: Enable flexible experimentation and reproducible runs primarily through configuration files, minimizing code changes.
- **Robustness & Futureproofing**: Aim for a stable, well-tested codebase with clear interfaces and adaptable design to facilitate long-term development and integration of new techniques.
- **Usability**: Offer a clear command-line interface (CLI) and well-defined workflows for common tasks like training, generation, and data preparation.

## Guiding Principles

- **Adherence to Conventions**: This project strives to follow established best practices and common conventions within the Python and ML ecosystems. Deviations are made thoughtfully, primarily when necessitated by critical performance requirements or essential custom implementations not suitably addressed by standard approaches.

## Overview

Craft provides a flexible architecture for experimenting with different model types, training methods, and data sources. It is designed for rapid prototyping, research, and building complex training pipelines driven by configuration, with a focus on clear structure and validation.

## Features

- **Unified CLI**: Command-line interface (`python -m craft.cli.run ...`) powered by Typer for training, generation, data preparation, and potentially other tasks.
- **Modular Design**: Clean separation between models, data processing, training logic, and configuration.
- **Configuration System**:
    - Uses [Hydra](https://hydra.cc/) for composing configurations from YAML files (`conf/`).
    - Defines configuration *schemas* using [Pydantic](https://docs.pydantic.dev/) (`src/craft/config/schemas.py`) for strong typing, validation, and clear structure.
    - Key components (models, optimizers, datasets, etc.) are automatically instantiated by Hydra based on configuration (`_target_`) and validated against Pydantic schemas.
- **Centralized Training**: The `Trainer` class (`src/craft/training/trainer.py`) orchestrates the training loop, evaluation, checkpointing, and callbacks based on the provided configuration.
- **Extensible Architecture**: Easily add new model types, datasets, tokenizers, callbacks, and training methods by implementing required interfaces and updating configurations.
- **Checkpoint Management**: Built-in utilities for saving and loading model checkpoints, including optimizer/scheduler state, training state, and tokenizer info.
- **Performance**: Supports features like Automatic Mixed Precision (AMP) via configuration.

## Core Architectural Components

The project is organized into several key directories:

-   **`src/craft/`**: The core library code.
    -   `config/`: Contains `schemas.py`, defining Pydantic models for configuration validation.
    -   `models/`: Implementations of different neural network architectures (e.g., `transformer.py`, `simple_rnn.py`). Models typically inherit from base classes defined here.
    -   `data/`: Data loading (`datasets/`), tokenization (`tokenizers/`), and related utilities. Includes base classes and concrete implementations (e.g., `SentencePieceTokenizer`, `PickledDataset`).
    -   `training/`: Core training logic (`trainer.py`, `training_loop.py`), evaluation (`evaluation.py`), checkpointing (`checkpointing.py`), callbacks (`callbacks/`), sampling (`sampling.py`), and generation utilities (`generation.py`).
    -   `cli/`: Logic for the command-line interface (`run.py`) using Typer.
    -   `utils/`: Common helper functions (logging, I/O, etc.).
-   **`conf/`**: Central hub for all [Hydra](https://hydra.cc/) configurations in YAML format. See the [conf/README.md](conf/README.md) for details on structure and composition.
    -   Organized into groups: `experiment/`, `model/`, `data/`, `optimizer/`, `scheduler/`, `training/`, `callbacks/`, etc.
    -   Relies heavily on the `_target_` key to link configurations to specific Python classes/functions for instantiation. The parameters defined here are validated against the Pydantic schemas in `src/craft/config/schemas.py`.
-   **`scripts/`**: Standalone Python scripts, potentially used for auxiliary tasks or development purposes (e.g., `check_model_params.py`). **Note:** Core training, generation, and data prep should primarily use the CLI (`python -m craft.cli.run ...`). This folder may contain legacy or utility scripts. See `scripts/README.md`.
-   **`tests/`**: Contains unit and integration tests using `pytest`. Tests often leverage specific configuration files.
-   **`outputs/`**: Default directory where Hydra saves outputs of runs (logs, checkpoints, configs), organized by date/time or custom experiment names based on Hydra configuration.
-   **`data/`**: (Top-level, optional) Can be used for storing raw or processed datasets. Not tracked by default.

## Key Development & Usage Workflows

Interacting with the framework typically involves one or more of these workflows:

1.  **Data Preparation:**
    *   Convert raw data into a format suitable for a specific `Dataset` class (e.g., tokenized sequences saved in `.pkl` files).
    *   Train a tokenizer if necessary (e.g., using SentencePiece directly or a helper script if provided).
    *   Use the CLI `dataset prepare` command:
        ```bash
        # Example for character-level processing
        python -m craft.cli.run dataset prepare --input <raw_file> --output-dir <processed_dir> --tokenizer-type char

        # Example for subword (requires trained tokenizer & config)
        # Assuming conf/data/my_subword_data.yaml exists and points to the trained tokenizer model
        python -m craft.cli.run dataset prepare --input <raw_file> --output-dir <processed_dir> --config conf/data/my_subword_data.yaml
        ```
    *   Ensure a corresponding data configuration (`conf/data/*.yaml`) exists, defining the `_target_` (e.g., `craft.data.dataset.PickledDataset`), tokenizer details (if applicable), and data file paths.

2.  **Model Development:**
    *   Implement a new `torch.nn.Module` in `src/craft/models/`, often inheriting from a base class like `BaseModel` or `LanguageModel`.
    *   Define a Pydantic schema for its configuration in `src/craft/config/schemas.py` (nested within `AnyModelConfig`).
    *   Create a configuration file in `conf/model/` specifying the `_target_` (path to the new model class) and its hyperparameters.

3.  **Experiment Configuration:**
    *   Define a new experiment in `conf/experiment/`. An experiment file typically uses a `defaults` list to compose configurations from `model`, `data`, `optimizer`, `training`, etc.
    *   Override specific parameters as needed for the experiment (e.g., `training.learning_rate=5e-5`).

4.  **Training:**
    *   Launch training using the CLI `train` command, specifying the desired experiment configuration:
        ```bash
        python -m craft.cli.run train experiment=my_experiment [optional overrides...]
        # Example Override: python -m craft.cli.run train experiment=my_transformer training.num_epochs=10
        ```
    *   The `Trainer` class will be instantiated, components will be created via Hydra based on the composed config, and the training loop will commence.
    *   Monitor logs and artifacts saved by Hydra in the `outputs/` directory.

5.  **Inference/Generation:**
    *   Use a trained model checkpoint (`.pt` file) for tasks like text generation.
    *   Use the CLI `generate text` command, providing the checkpoint path and desired generation parameters:
        ```bash
        # Find your checkpoint (e.g., latest.pt or ckpt_step_X.pt) in the outputs/run_dir/checkpoints/
        python -m craft.cli.run generate text --checkpoint <path/to/checkpoint.pt> --prompt "Your starting prompt" [options...]
        # Example options: --max-new-tokens 100 --temperature 0.7
        ```

## Getting Started

1.  **Setup Environment:** Create and activate a Python environment (e.g., using `conda` or `venv`). Python 3.10+ is recommended (check `pyproject.toml` for specific constraints).
2.  **Installation:**
    *   **For users/running:**
        ```bash
        # Install exact dependencies from the lock file
        pip install -r requirements.lock.txt
        # Install the craft package itself
        pip install .
        # Or in editable mode if you might modify the source:
        # pip install -e .
        ```
    *   **For development (includes testing, linting tools):**
        ```bash
        # Install exact dependencies (including dev tools) from the dev lock file
        pip install -r requirements-dev.lock.txt
        # Install the craft package in editable mode
        pip install -e .
        ```
3.  **Prepare Data:** (Example using sample data provided in `data/raw/sample.txt`)
    ```bash
    # Create processed data directories if they don't exist
    mkdir -p data/processed/sample_char data/processed/sample_subword

    # Prepare character-level dataset
    python -m craft.cli.run dataset prepare --input data/raw/sample.txt --output-dir data/processed/sample_char --tokenizer-type char

    # (Optional) Prepare subword dataset (requires training SentencePiece first)
    # 1. Train SentencePiece model (using spm directly or a script if available)
    #    Example command if using spm directly:
    #    pip install sentencepiece # If not already installed
    #    spm_train --input=data/raw/sample.txt --model_prefix=data/processed/sample_sp --vocab_size=100 --model_type=bpe
    # 2. Use a data config (e.g., conf/data/subword_sample.yaml) pointing to the trained model
    #    Ensure conf/data/subword_sample.yaml exists and contains:
    #    tokenizer:
    #      _target_: craft.data.tokenizers.SentencePieceTokenizer
    #      model_path: data/processed/sample_sp # Path prefix to the trained model
    #    # ... other dataset config ...
    # python -m craft.cli.run dataset prepare --input data/raw/sample.txt --output-dir data/processed/sample_subword --config conf/data/subword_sample.yaml
    ```
4.  **Configure Experiment:** Review configurations in `conf/`. The provided `conf/experiment/char_small.yaml` should work with the character data prepared above. Ensure data paths match if you used different locations.
5.  **Run Training:**
    ```bash
    # This uses the configuration defined in conf/experiment/char_small.yaml
    python -m craft.cli.run train experiment=char_small
    ```
6.  **Generate Text:** After training completes, find the checkpoint in the `outputs/` directory (Hydra creates a dated subdirectory).
    ```bash
    # Replace <run_dir> with the actual directory created by Hydra (e.g., outputs/YYYY-MM-DD/HH-MM-SS/)
    python -m craft.cli.run generate text --checkpoint outputs/<run_dir>/checkpoints/latest.pt --prompt "The meaning of life is"
    ```

## Testing

Run tests using `pytest`:
```bash
pytest
```

## TODO

(Add any planned features or known issues here)