# Craft

A modular framework for building, training, and evaluating AI models, with a current focus on character-level language modeling.

## Overview

Craft provides a flexible architecture for experimenting with different model types, training methods, and data sources. It is particularly well-suited for research and rapid prototyping of new ideas. While initially specialized in language models like the character-level transformer for Game of Thrones-style text generation (ChatGoT), the framework is designed to be extended to other model types.

## Features

- **Unified CLI**: Intuitive command-line interface for training, generation, and experiments
- **Modular Design**: Clean separation between models, data processing, and training
- **Configuration-Driven**: YAML-based configuration for experiments without code changes
- **Optimized Training**: Performance optimizations for efficient training
- **Checkpoint Management**: Built-in utilities for saving and loading model checkpoints
- **Extensible Architecture**: Easily add new model types, datasets, and training methods

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/craft.git
cd craft

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Quick Start

### Command Line Interface

Craft provides a unified command-line interface for all operations:

```bash
# Training a language model
python scripts/craft train language --config conf/experiments/char_transformer.yaml

# Generating text with a trained model
python scripts/generate_samples.py --model_path outputs/models/best_model.pt --prompt "The king"

# Preparing data for training
python scripts/craft dataset prepare --input data/raw/got.txt --output-dir data/processed

# Running a predefined experiment
python scripts/craft experiment run --config conf/experiments/experiment1.yaml
```

### Python API

You can also use Craft as a Python library:

```python
from src.models import create_model_from_config
from src.data import create_data_manager
from src.training import create_trainer_from_config
from src.utils import load_experiment_config

# Load configuration
config = load_experiment_config("conf/experiments/char_transformer.yaml")

# Create data manager and prepare data
data_manager = create_data_manager(config["data"])
train_loader, val_loader = data_manager.prepare_dataloaders("text", split=["train", "val"])

# Create model
model = create_model_from_config(config["model"])

# Create trainer
trainer = create_trainer_from_config(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    config=config["training"]
)

# Train model
metrics = trainer.train()

# Generate text
generated_text = model.generate("In the name of", max_length=100)
print(generated_text)
```

## Project Structure

```
craft/
├── conf/                   # Configuration files (using Hydra/OmegaConf)
│   ├── data/               # Data configurations
│   ├── experiment/         # Combined experiment configurations
│   ├── model/              # Model configurations
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

For a more detailed explanation of the directory structure, see [Architecture Documentation](docs/architecture.md#directory-structure).

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Getting Started Guide](docs/guides/getting_started.md)
- [Architecture Overview](docs/architecture.md)
- [Extending Craft](docs/extending.md)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.