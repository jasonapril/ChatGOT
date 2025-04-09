# Craft

A modular PyTorch framework designed for building, training, and evaluating AI models, leveraging Hydra for configuration and Pydantic for validation.

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
- **Visualization**: Includes a `TensorBoardLogger` callback to log training and validation metrics (loss, learning rate, throughput, perplexity, etc.). Logs are typically saved to `outputs/<experiment_name>/<run_id>/tensorboard_logs/`. Use `tensorboard --logdir outputs/` (or a more specific run directory) to view plots.

## Project Goals

- **Modularity & Extensibility**: Provide a clean architecture that allows easy addition and swapping of components like models, datasets, tokenizers, and training routines.
- **Configuration Driven**: Enable flexible experimentation and reproducible runs primarily through configuration files, minimizing code changes.
- **Robustness & Futureproofing**: Aim for a stable, well-tested codebase with clear interfaces and adaptable design to facilitate long-term development and integration of new techniques.
- **Usability**: Offer a clear command-line interface (CLI) and well-defined workflows for common tasks like training, generation, and data preparation.
- **Performance & Resource Efficiency**: Optimize for fast training and inference times while maximizing the utilization of available system resources (RAM/VRAM) within configurable limits, particularly targeting effectiveness on low-spec hardware.
- **Experimentation & Innovation**: Facilitate the rapid development, testing, and iteration of novel AI architectures and techniques (including custom ideas and published research) with a focus on discovering efficient and effective methods suitable for various hardware constraints. Enable modification and use of third-party models where feasible.

## Guiding Principles

- **Adherence to Conventions**: This project strives to follow established best practices and common conventions within the Python and ML ecosystems. Deviations are made thoughtfully, primarily when necessitated by critical performance requirements or essential custom implementations not suitably addressed by standard approaches.
- **Workflow-Driven Development**: Development focuses on establishing and refining working end-to-end workflows (e.g., data preparation, training, generation). Components (models, data loaders, tests, etc.) are built and improved incrementally to support and expand these core functional pipelines. This approach prioritizes functional integration and robust core capabilities.

### Dependency Management Guidelines

Adding external dependencies increases complexity and potential risks. Before adding a new dependency, please consider the following:

1.  **Necessity First**: Can the desired functionality be achieved reasonably using existing project dependencies or Python's standard library?
2.  **Value Assessment**: Does the library offer substantial benefits (unique functionality, significant performance improvement, major development time savings) that justify adding it?
3.  **Reputation & Maintenance**: Is the library well-maintained, actively developed, and widely used or recognized within its domain? Check for recent activity, open issues, and documentation quality.
4.  **License Compatibility**: Ensure the library's license (e.g., MIT, Apache 2.0, BSD) is permissive and compatible with the project's usage and potential future distribution.
5.  **Minimalism**: If alternatives exist, prefer libraries that are more focused or introduce fewer transitive dependencies, unless a heavier option provides significantly more value.
6.  **Security**: Be mindful of potential security implications. Check for known vulnerabilities if possible (e.g., using `pip-audit` or GitHub's Dependabot).

## Core Technology Rationale

Craft leverages several key frameworks and libraries, chosen for specific benefits:

- **PyTorch**: Selected as the core deep learning framework for its flexibility, strong community support, extensive ecosystem, and Pythonic design, facilitating rapid research and development.
- **Hydra**: Used for configuration management. Its ability to compose configurations from YAML files (`conf/`).
- **Pydantic**: Employed for defining configuration schemas (`src/craft/config/schemas.py`). It provides data validation, type hints, and clear settings documentation, enhancing robustness and ensuring configuration integrity before potentially expensive runs.
- **Typer**: Powers the command-line interface (CLI). Chosen for its ease of use, automatic help generation, and seamless integration, making interactions with the framework straightforward.

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
-   **`outputs/`**: Default directory where Hydra saves outputs of runs, organized by experiment and run ID (e.g., `outputs/experiments/<experiment_name>/<run_id>/`). See "Development Standards & Guidelines" for details on artifact colocation (checkpoints, logs).
-   **`data/`**: (Top-level, optional) Can be used for storing raw or processed datasets. Not tracked by default.

## Development Standards & Guidelines

- **PEP 8 Compliance**: All Python code should adhere to the PEP 8 style guide.
- **Single Source of Truth (SSOT)**: Documentation should follow the SSOT principle. Information relevant to a specific component (e.g., configuration details) should ideally reside close to that component (e.g., in its `README.md` or docstrings), while this main `README.md` provides the high-level overview and project status.
- **Maintainability & Debuggability**: Code should be written clearly, be well-documented (docstrings, comments for non-obvious logic), and accompanied by comprehensive tests to facilitate understanding, modification, and troubleshooting.
- **Experiment Artifact Colocation**: Hydra outputs (logs, checkpoints, etc.) should be organized per experiment. The standard structure within the `outputs/experiments/` directory is `<experiment_name>/<run_id>/` (where `<run_id>` is typically `YYYY-MM-DD/HH-MM-SS/` by default), with checkpoints specifically saved to `outputs/experiments/<experiment_name>/<run_id>/checkpoints/` and logs potentially to `outputs/experiments/<experiment_name>/<run_id>/logs/`. This structure makes finding artifacts for a specific run straightforward.
- **Pre-Commit Checks**: Use `mypy src` for type checking and `pytest` to run tests before committing changes.

## Key Development & Usage Workflows

Interacting with the framework typically involves one or more of these workflows:

1.  **Data Preparation:**
    *   Convert raw data into a format suitable for a specific `Dataset` class (e.g., tokenized sequences saved in `.pkl` files).
    *   Train a tokenizer if necessary (e.g., using SentencePiece directly or the `dataset train-tokenizer` CLI command).
    *   Use the CLI `dataset prepare` command:
        ```bash
        # Example for character-level processing
        # Uses default splits (0.9, 0.05, 0.05)
        python -m craft.cli.run data prepare --input-path data/raw/sample.txt --output-dir data/processed/sample_char --type char

        # Example for subword processing (requires trained tokenizer)
        # Assumes a tokenizer model exists at e.g., outputs/tokenizers/my_spm.model
        python -m craft.cli.run data prepare --input-path data/raw/sample.txt --output-dir data/processed/sample_subword --type subword --tokenizer-path outputs/tokenizers/my_spm.model --split-ratios 0.9,0.05,0.05
        ```
    *   Ensure a corresponding data configuration (`conf/data/*.yaml`) exists for use during *training*, defining the `_target_` (e.g., `craft.data.datasets.PickledDataset`), the path to the *processed* data file (e.g., `data_path: data/processed/sample_subword/train.pkl`), and `block_size`.

2.  **Model Development:**
    *   Implement a new `torch.nn.Module` in `src/craft/models/`, often inheriting from a base class like `BaseModel` or `LanguageModel`.
    *   Define a Pydantic schema for its configuration in `src/craft/config/schemas.py` (nested within `AnyModelConfig`).
    *   Create a configuration file in `conf/model/` specifying the `_target_` (path to the new model class) and its hyperparameters.

3.  **Adding New Model Architectures:** To integrate a new model architecture (`MyNewModel`) into the framework's configuration system:
    *   **Implement the Model:** Create your model class (e.g., `MyNewModel`) in a suitable file within `src/craft/models/`, inheriting from `BaseModel` or another appropriate base.
    *   **Define Pydantic Schema:** In `src/craft/config/schemas.py`:
        *   Define a new schema class (e.g., `MyNewModelConfig`) inheriting from `BaseModelConfig` (or a more specific base like `GenerativeModelConfig`).
        *   Add an `architecture` field with a unique `Literal` string value (e.g., `architecture: Literal["my_new_model"] = "my_new_model"`). This string is the discriminator.
        *   Define other necessary configuration parameters for your model within this schema.
    *   **Update Union Type:** Add your new schema class (`MyNewModelConfig`) to the `_ModelConfigUnion` type alias in `src/craft/config/schemas.py`.
    *   **Create Hydra Config:** Create a YAML file (e.g., `conf/model/my_new_model.yaml`) with:
        *   `_target_`: The full Python path to your model class (e.g., `craft.models.my_new_model_file.MyNewModel`).
        *   `architecture`: The unique string literal you defined in the schema (e.g., `my_new_model`).
        *   Other hyperparameters corresponding to the fields in your `MyNewModelConfig` schema.
    *   You can now reference this model configuration in your experiment files (e.g., `conf/experiment/my_new_model_exp.yaml`) using `model: my_new_model` in the `defaults` list.

4.  **Experiment Configuration:**
    *   Define a new experiment in `conf/experiment/`. An experiment file typically uses a `defaults` list to compose configurations from `model`, `data`, `optimizer`, `training`, etc.
    *   Override specific parameters as needed for the experiment (e.g., `training.learning_rate=5e-5`).

5.  **Training:**
    *   Launch training using the CLI `train` command, specifying the desired experiment configuration:
        ```bash
        python -m craft.cli.run train experiment=my_experiment [optional overrides...]
        # Example Override: python -m craft.cli.run train experiment=my_transformer training.num_epochs=10
        ```
    *   The `Trainer` class will be instantiated, components will be created via Hydra based on the composed config, and the training loop will commence.
    *   Monitor logs and artifacts saved by Hydra in the `outputs/` directory.

6.  **Inference/Generation:**
    *   Use a trained model checkpoint (`.pt` file) for tasks like text generation.
    *   Use the CLI `generate text` command, providing the checkpoint path and desired generation parameters:
        ```bash
        # Find your checkpoint (e.g., latest.pt or ckpt_step_X.pt) in the outputs/run_dir/checkpoints/
        python -m craft.cli.run generate text --checkpoint <path/to/checkpoint.pt> --prompt "Your starting prompt" [options...]
        # Example options: --max-new-tokens 100 --temperature 0.7
        ```
        
## TODO

### Active

-   **Define and Implement Core Functionality & Testing Roadmap (Priority: Critical):** Ensure the project robustly supports the following key workflows, with comprehensive tests for each. Align all code and documentation with these priorities.
    -   *Recent Progress:* Completed refactoring of core components (`Trainer`, `TrainingLoop`) for `TrainingConfig` integration and dependency simplification. Refactored `TrainingLoop` to handle gradient accumulation, AMP (scaler usage), optimizer steps, gradient clipping, callbacks (`on_step_begin`, `on_step_end`), and error handling correctly. Unit tests for `TrainingLoop` (`tests/training/training_loop/`) extensively revised and corrected to pass after refactoring. Addressed related unit test failures, including checkpoint resuming (`Trainer._resume_from_checkpoint`). (DONE)
    -   **(Completed) Data Preparation:** Implemented and tested the `data prepare` CLI command for `char` and `subword` types. `PickledDataset` now loads data and automatically handles metadata (including tokenizer info) from `metadata.json` in the processed data directory. Added CLI tests.
    -   **(Completed) Model Configuration & Development:** Solidified and documented the process using Pydantic schemas (`src/craft/config/schemas.py`) and Hydra configs (`conf/model/`). Confirmed alignment via discriminator pattern and added documentation for adding new architectures. Verified existing tests in `tests/config/test_schemas.py` cover configuration validation and discriminator logic.
    -   **Training Workflow:** Refine and test the `train` CLI command, the `Trainer` class (`src/craft/training/trainer.py`), and methods for setting up training runs.
        -   *Logging:* Ensure essential metrics (throughput T/s, VRAM/CUDA utilization, loss) are logged correctly. Evaluate integration with tools like TensorBoard (See `## Review Dependency Rationale`). Ensure logs are directed appropriately (e.g., `outputs/<experiment_name>/`). (Completed VRAM/Tput calculation in TrainingLoop, logged via TensorBoard if enabled).
        -   *Checkpointing:* Thoroughly test saving (`checkpointing.py`) and resuming (`Trainer._resume_from_checkpoint`) training state (model weights, optimizer state, scheduler state, training progress). Test edge cases and recovery. Ensure checkpoints are saved to logical locations (e.g., `outputs/<experiment_name>/checkpoints/`). Added integration tests for basic, latest, AMP, and scheduler resume scenarios (`tests/integration/test_checkpoint_resume.py`). (DONE)
        -   *Sample Generation during Training:* Refactored sample generation to be callback-driven (`SampleGenerationCallback`) and added tests (`tests/training/callbacks/test_sample_generation.py`). Removed time-based triggers from `TrainingLoop`. (DONE)
        -   *Refactoring:* Extracted component initialization from `Trainer` into `initialization.py` to improve clarity and maintainability. (DONE)
    -   **Evaluation & Analysis:** Develop and test methods (`evaluation.py`) for evaluating models post-training. Implement loss visualization tools/scripts.
        -   *`evaluation.py` Module:* Existing `Evaluator` reviewed. `Evaluator` integrated into `Trainer`. Perplexity calculation added to `Evaluator.evaluate`. (DONE)
        -   *Add Tests for `Evaluator`*: Added tests in `tests/training/test_evaluation.py`. (DONE)
        -   *Loss Visualization Tools/Scripts:* DONE (via TensorBoard)
    -   **Experimentation & Config:** (DONE)
        -   *Hydra Sweeps & Output Structure (`outputs/experiments/...`):* DONE
        -   *Configuration Schema Refinement (Pydantic):* DONE
    -   **Advanced Features:** (Starting)
        -   *Gradient Accumulation:* Functionality implemented and tested within `TrainingLoop`. (DONE)
        -   *Distributed Training:* TODO
        -   *Quantization/Pruning Hooks:* TODO
    -   **End-to-End Integration Tests:** Expand tests like `tests/integration/test_transformer_pipeline.py`. Add CLI-based integration tests (e.g., in `tests/cli/integration/`) to cover full workflows (data prep -> train -> evaluate/generate). Added `test_checkpoint_resume.py`.
-   Address any remaining integration test failures identified during full test suite runs.
-   **(Meta-Rule):** Keep this section updated so that we could start a new chat without context and be able to resume what we were working on.

### Misc

### Review Configuration System

- Explore deeper integration with Hydra's Structured Configs (using dataclasses).

### Review Dependency Rationale

- Mention that this project uses mypy? Document *why* we're using it?
- Review role of Pydantic. Is integration with Hydra conventional?
- Are we using TensorBoard? Why or why not? Ensure logs go to outputs/<experiment_name>/

### Review Testing

- Are we using any smoke tests?
- Can we design tests around features or model development phases like data prep, model architecture design, training, evaluation, analysis, inference
- Improve test coverage.
- Expand integration testing.
- Address TODOs and FIXMEs in the code.
- Standardize docstrings (NumPy style?)

### Meta

- **Refactor this README.md (High Priority)** -- it contains some redundant information. Review how this document is structured including names and order of headings. Note that this document is to help an AI Agent understand the project at a glance (in addition to the other README.md files, local to subfolders), for the time being, at least.
- Is there a better location for documentation like this?
- Would it be helpful to link to larger sets of tasks, projects, plans, or roadmaps from this document for the AI Agent to use as a sort of long-term memory?

### Experiments

**ChatGOT-char**: Train 1M, 10M, and 100M parameter models on GOT scripts using character-level tokens.

**ChatGOT-subword**: Train 1M, 10M, and 100M parameter models on GOT scripts using subword-level tokens.

### Feature Enhancements

**Add Support for Different Model Architectures**: Diffusion, RNN, etc.

**Add Ability to Train on MNIST**
- This marks a major transition to multimodality. This will also ensure a higher degree of modularity.

**Add Support for Third-Party Models**
