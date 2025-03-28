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
  - `BaseModel`: The core base class for all models.
  - `GenerativeModel`: For models that generate outputs.
  - `LanguageModel`: For language models specifically.
- `char_lstm.py`: A character-level LSTM language model.
- `transformer.py`: A transformer-based language model.

Models follow an object-oriented design with clear inheritance hierarchies to ensure consistent interfaces.

### Data

The `src/data/` directory handles data loading, processing, and dataset creation:

- `base.py`: Contains abstract base classes for different dataset types.
  - `BaseDataset`: The core base class for all datasets.
  - `TextDataset`: For text-based datasets.
- `char_dataset.py`: A character-level text dataset.
- `loaders.py`: Functions for creating data loaders.
- `processors.py`: Functions for processing different types of data (text, image, etc.).
- `tokenization.py`: Character-level tokenization utilities.

### Training

The `src/training/` directory contains training loops and evaluation code:

- `base.py`: Contains abstract base classes for different trainer types.
  - `Trainer`: The core base class for all trainers.
  - `LanguageModelTrainer`: For training language models.
- `evaluation.py`: Evaluation metrics and utilities.
- `optimization.py`: Optimization utilities.
- `logging.py`: Training loggers for tracking experiments.

### Configuration

The `configs/` directory contains configuration files for models, datasets, training, and experiments:

- `models/`: Model configuration files.
- `data/`: Dataset configuration files.
- `training/`: Training configuration files.
- `experiments/`: Combined configuration files for complete experiments.

Configuration files are YAML-based and can be composed to create complex experimental setups.

### Utils

The `src/utils/` directory contains general utilities:

- `common.py`: Common utilities for setup, seeding, and memory tracking.
- `config.py`: Configuration loading and parsing utilities.
- `logging.py`: Logging utilities.
- `visualization.py`: Visualization utilities for model outputs.

## Workflow

A typical workflow in Craft looks like this:

1. **Configuration**: Define or select configurations for your model, data, and training.
2. **Dataset Creation**: Create a dataset from the data configuration.
3. **Model Creation**: Create a model from the model configuration.
4. **Trainer Creation**: Create a trainer from the training configuration.
5. **Training**: Train the model using the trainer.
6. **Evaluation**: Evaluate the model on test data.
7. **Inference**: Use the trained model for inference.

This workflow is typically orchestrated through a command-line interface defined in `src/cli/`.

## Directory Structure

```
craft/
├── configs/                # Configuration files
│   ├── data/               # Data configurations
│   ├── experiments/        # Combined experiment configurations
│   ├── models/             # Model configurations
│   └── training/           # Training configurations
├── data/                   # Data storage
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data
├── docs/                   # Documentation
│   ├── guides/             # User guides
│   ├── architecture.md     # Architecture overview
│   └── extending.md        # Extending the framework
├── flow/                   # Task management system
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
├── scripts/                # Scripts and entry points
├── src/                    # Source code
│   ├── cli/                # Command-line interface
│   ├── config/             # Configuration management
│   ├── data/               # Data loading and processing
│   ├── models/             # Model definitions
│   ├── training/           # Training loops and evaluation
│   └── utils/              # Utilities
└── tests/                  # Unit tests
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
3. **Configuration**: Config files should go in `configs/`.
4. **Documentation**: Documentation files should go in `docs/`.
5. **Planning**: Project management documents go in `flow/`. 