# Craft Architecture

This document provides an overview of the Craft architecture and explains how the different components interact.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
  - [Models](#models)
  - [Data](#data)
  - [Training](#training)
  - [Configuration](#configuration)
  - [Utils](#utils)
- [Workflow](#workflow)
- [Directory Structure](#directory-structure)

## Overview

Craft is a framework for building, training, and evaluating language models and other generative models. It is designed to be modular, extensible, and configurable, allowing you to easily experiment with different model architectures, training methods, and data sources.

## Core Components

### Models

The `src/models/` directory contains model definitions and related utilities:

- `base.py`: Contains abstract base classes for different model types.
  - `Model`: The core base class for all models (renamed from BaseModel).
  - `GenerativeModel`: For models that generate outputs.
  - `LanguageModel`: For language models specifically.
- `gpt_decoder.py`: A custom GPT-like decoder implementation.
- `transformer.py`: A transformer-based language model (potentially using `nn.TransformerDecoder`).
- `__init__.py`: Makes the directory a Python package and potentially exposes key classes.

Models follow an object-oriented design with clear inheritance hierarchies to ensure consistent interfaces.

### Data

The `src/data/` directory handles data loading, processing, and dataset creation:

- `base.py`: Contains abstract base classes and core data loading functions (e.g., `prepare_dataloaders_from_config`).
  - `BaseDataset`: The core base class for all datasets (may reside here or in `dataset.py`).
  - `TextDataset`: For text-based datasets (may reside here or in `dataset.py`).
- `dataset.py`: Contains specific `torch.utils.data.Dataset` implementations (e.g., `CharDataset` might be here if still used).
- `processors.py`: Functions for processing different types of data (text, image, etc.).
- `__init__.py`: Makes the directory a Python package.

*(Note: `loaders.py` and `tokenization.py` are no longer present in this directory.)*

### Training

The `src/training/` directory contains training loops, callbacks, and related utilities:

- `base.py`: Contains the core `Trainer` base class (and potentially implementations like `LanguageModelTrainer`).
- `callbacks.py`: Contains `TrainerCallback` base class and specific callback implementations (e.g., `ReduceLROnPlateauOrInstability`, `SampleGenerationCallback`, `TensorBoardLogger`).
- `amp.py`: Utilities for Automatic Mixed Precision (AMP) training, including `SafeGradScaler`.
- `utils.py`: Training-specific utility functions (e.g., `enable_gradient_checkpointing`).
- `__init__.py`: Makes the directory a Python package.

*(Note: The following files might represent older or redundant training logic and may be candidates for removal/refactoring: `evaluation.py`, `generation.py`, `optimizations.py`, `train_config.py`, `train_runner.py`, `training_loop.py`. The primary training flow should now be centered around `base.py` and `callbacks.py`.)*

### Config Management (`src/config/`)

The `src/config/` directory contains code related to loading, validating, and managing configurations (distinct from the configuration *files* in `conf/`):

- `config_manager.py`: Handles loading, merging, and potentially validating configuration objects (e.g., using Hydra/OmegaConf).
- `__init__.py`: Makes the directory a Python package.

### Command-Line Interface (`src/cli/`)

The `src/cli/` directory provides the command-line interface for interacting with the framework:

- `run.py`: Defines the main CLI application entry point (e.g., using Typer or Click) and orchestrates workflows like training, evaluation, and generation based on commands and configurations.
- `__init__.py`: Makes the directory a Python package.

### Configuration

The `conf/` directory contains configuration files for models, datasets, training, and experiments:

- `models/`: Model configuration files.
- `data/`: Dataset configuration files.
- `training/`: Training configuration files.
- `experiments/`: Combined configuration files for complete experiments.

Configuration files are YAML-based and can be composed to create complex experimental setups.

### Utils

The `src/utils/` directory contains general utilities shared across the project:

- `common.py`: Common utilities for setup (e.g., `set_seed`), device handling (`setup_device`).
- `logging.py`: Logging setup utilities (`setup_logger`).
- `io.py`: Input/output utilities.
- `checkpoint.py`: Functions for saving and loading model checkpoints (`save_checkpoint`, `load_checkpoint`).
- `performance.py`: Utilities for measuring performance (e.g., `get_resource_metrics`).
- `metrics.py`: Standalone metric calculation functions.
- `memory.py`: Memory usage tracking utilities.
- `generation.py`: Potentially contains standalone text generation helper functions (separate from model's own generate method or training callbacks).
- `__init__.py`: Makes the directory a Python package and may expose key utilities.

*(Note: `config.py` and `visualization.py` are no longer present in this directory. Configuration handling is now in `src/config/`.)*

## Workflow

A typical workflow in Craft looks like this:

1. **Configuration**: Define or select configurations in the `conf/` directory (e.g., for model, data, training) or override via command line.
2. **Data Preparation**: Use `src/data/` components (potentially via CLI) to process raw data.
3. **Model Initialization**: The `src/models/` classes are instantiated based on configuration.
4. **Trainer Setup**: The `src/training/base.py::Trainer` (or a subclass) is set up with the model, data loaders, optimizer, scheduler, and callbacks.
5. **Training Execution**: Run training using the `Trainer`'s `train()` method, often triggered by a command in `src/cli/run.py`.
6. **Evaluation**: Evaluate the model using the `Trainer`'s `evaluate()` method or standalone evaluation scripts.
7. **Inference/Generation**: Use the trained model for inference, potentially via a generation command in `src/cli/run.py` or by loading the checkpoint and using the model's `generate()` method directly.

This workflow is typically orchestrated through the command-line interface defined in `src/cli/run.py` using configuration files managed by `src/config/config_manager.py`.

## Directory Structure

```
craft/
├── conf/                 # Configuration files (YAML format)
│   ├── callbacks/          # Callback configurations
│   ├── data/               # Data configurations
│   ├── experiment/         # Combined experiment configurations
│   ├── model/              # Model configurations (Singular)
│   └── training/           # Training configurations
├── data/                   # Data storage (managed outside source control)
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data
├── docs/                   # Documentation
│   ├── guides/             # User guides
│   ├── architecture.md     # Architecture overview
│   └── extending.md        # Extending the framework
├── flow/                   # Task management & project context
│   ├── active/             # Current work and tasks
│   ├── logs/               # Project logs and records
│   ├── meta/               # Meta-documentation
│   ├── performance/        # Performance analysis
│   ├── planning/           # Project planning
│   ├── reference/          # Reference materials
│   └── system/             # System-related configurations
├── models/                 # Legacy saved models (use outputs/models for new models)
├── outputs/                # All generated files and artifacts
│   ├── benchmarks/         # Performance benchmark results
│   ├── evaluation/         # Model evaluation results
│   ├── logs/               # Training and evaluation logs
│   ├── models/             # Trained model checkpoints
│   ├── samples/            # Generated text samples
│   └── visualizations/     # Plots and charts
├── scripts/                # Standalone scripts (utility, testing, legacy)
├── src/                    # Source code
│   ├── cli/                # Command-line interface (run.py)
│   ├── config/             # Configuration management code (config_manager.py)
│   ├── data/               # Data loading and processing (base.py, dataset.py, processors.py)
│   ├── models/             # Model definitions (base.py, gpt_decoder.py, transformer.py)
│   ├── training/           # Training loops and evaluation (base.py, callbacks.py, amp.py, utils.py, legacy?)
│   └── utils/              # Shared utilities (common.py, io.py, checkpoint.py, logging.py, etc.)
└── tests/                  # Unit and integration tests
```

The structure is designed to separate concerns and make it easy to navigate and extend the codebase.

### Source Code (`src/`)

Contains the main implementation of models, data processing, utilities, and training code.

### Outputs (`outputs/`)

Contains all files generated by running the code. This is the standard location for all outputs:

- **models/**: Stores model checkpoints generated during training
  - Format: `{model_name}_{size}_{precision}_{date}_{timestamp}_step_{step}.pt`
  
- **logs/**: Contains logs from training and evaluation runs
  - Training logs: `train_{model_name}_{date}_{timestamp}.log`
  - Evaluation logs: `eval_{model_name}_{date}_{timestamp}.log`
  
- **samples/**: Contains generated text samples
  - Format: `sample_{date}_{timestamp}.txt`
  
- **evaluation/**: Contains evaluation metrics and results
  - Format: `eval_{model_name}_{dataset}_{date}_{timestamp}.json`
  
- **visualizations/**: Contains generated charts and plots
  - Format: `{visualization_type}_{model_name}_{date}_{timestamp}.png`
  
- **benchmarks/**: Contains performance benchmark results
  - Format: `benchmark_{test_name}_{hardware}_{date}_{timestamp}.json`

### Flow (`flow/`)

Contains project planning, documentation, and tracking information.

### Usage Guidelines

1. **Code Development**: All source code should go in `src/`.
2. **Generated Files**: All outputs from running code should go in `outputs/`.
3. **Configuration**: Config files should go in `conf/`.
4. **Documentation**: Documentation files should go in `docs/`.
5. **Planning**: Project management documents go in `flow/`. 