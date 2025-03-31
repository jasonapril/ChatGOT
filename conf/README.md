# Configuration (`conf/`)

This directory contains configuration files for the Craft project, primarily managed by [Hydra](https://hydra.cc/).

## Structure:

*   `config.yaml`: The main Hydra configuration file. It defines the default composition of configuration groups and may contain top-level settings or Hydra-specific configurations (like logging and output directories).
*   Subdirectories (Configuration Groups): Each subdirectory typically represents a configurable component of the application (e.g., `model`, `data`, `training`). These directories contain YAML files that define specific options for that component (e.g., `conf/model/gpt_decoder.yaml`).
    *   `callbacks/`: Configurations for training callbacks.
    *   `data/`: Dataset configurations.
    *   `experiment/`: Pre-defined, complete experiment configurations that compose settings from various groups for specific runs (e.g., `chatgot_25m.yaml`). These are often selected via the command line (`python -m src.cli.run experiment=some_experiment`).
    *   `model/`: Model architecture configurations.
    *   `optimizer/`: Optimizer configurations.
    *   `scheduler/`: Learning rate scheduler configurations.
    *   `training/`: Training loop configurations.
    *   `tokenizer/`: (Potentially used, currently seems integrated elsewhere).
*   Other `.yaml` files: Standalone configuration files (e.g., `test_minimal.yaml` potentially for testing).

## Usage:

Configurations are typically loaded and merged by Hydra when running scripts like `src/cli/run.py`. Defaults are set in `config.yaml`, and specific settings can be chosen or overridden via the command line (e.g., `python -m src.cli.run model=gpt_decoder data=my_dataset training.epochs=5`). Experiment files provide convenient presets for common runs. 