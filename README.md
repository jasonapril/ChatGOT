# Craft

A modular framework for building, training, and evaluating AI models.

## Overview

Craft provides a flexible architecture for experimenting with different model types, training methods, and data sources. It is particularly well-suited for research and rapid prototyping of new ideas.

## Features

- **Unified CLI**: Intuitive command-line interface for training, generation, and experiments
- **Modular Design**: Clean separation between models, data processing, and training
- **Configuration-Driven**: YAML-based configuration for experiments without code changes
- **Performance Optimization**: Performance optimizations targeting efficient use of resources on standard hardware.
- **Checkpoint Management**: Built-in utilities for saving and loading model checkpoints
- **Extensible Architecture**: Easily add new model types, datasets, and training methods

## Project Structure
```
craft/
├── conf/       # Configuration files (using Hydra/OmegaConf)
├── data/       # Data storage
├── flow/       # System for providing AI agent with memory
├── notebooks/  # For experimentation, exploration, and visualization
├── outputs/    # All generated files and artifacts
├── scripts/    # For execution. Helper scripts and entry points
├── src/        # For library code. Main applicaton source code
└── tests/      # Tests (unit, integration, etc.)
```