# Getting Started with Craft

This guide will help you get started with Craft, a framework for building, training, and evaluating AI models with a current focus on language modeling.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Using the Command Line Interface](#using-the-command-line-interface)
- [Running Your First Experiment](#running-your-first-experiment)
- [Creating a Custom Configuration](#creating-a-custom-configuration)
- [Next Steps](#next-steps)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/craft.git
cd craft
```

2. Create and activate a virtual environment:

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n craft python=3.10
conda activate craft
```

3. Install the package:

```bash
pip install -e .
```

## Project Structure

The Craft project is organized with a clean separation of concerns:

```
craft/
├── conf/                # Configuration files
│   ├── data/               # Data configurations
│   ├── experiments/        # Combined configurations
│   ├── models/             # Model configurations
│   └── training/           # Training configurations
├── data/                   # Data storage
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data
├── models/                 # Saved models
├── scripts/                # Scripts and entry points
└── src/                    # Source code
    ├── cli/                # Command-line interface
    ├── config/             # Configuration management
    ├── data/               # Data loading and processing
    ├── models/             # Model definitions
    ├── training/           # Training loops and evaluation
    └── utils/              # Utilities
```

## Using the Command Line Interface

Craft provides a unified command-line interface for all operations. The main entry point is the `craft` script:

```bash
python scripts/craft --help
```

This will show you all available commands. The main commands are:

- `train`: Commands for model training
- `generate`: Commands for text generation
- `dataset`: Commands for dataset operations
- `experiment`: Commands for running experiments
- `evaluate`: Commands for model evaluation

Each command has subcommands. For example:

```bash
# Show train subcommands
python scripts/craft train --help

# Show language model training options
python scripts/craft train language --help

# Show text generation options
python scripts/craft generate text --help
```

## Running Your First Experiment

Let's train a character-level language model on a sample dataset.

### Step 1: Prepare Data

First, ensure you have some text data in the `data/raw/` directory:

```bash
mkdir -p data/raw
curl -o data/raw/sample.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Then, process the data using the CLI:

```bash
python scripts/craft dataset prepare --input data/raw/sample.txt --output-dir data/processed/shakespeare
```

### Step 2: Train a Model

Use the train command to train a language model:

```bash
python scripts/craft train language \
    --config conf/experiments/char_transformer.yaml \
    --output-dir models/shakespeare
```

This will start the training process. You'll see progress updates in the console, including:
- Training and validation loss
- Generation samples at regular intervals
- Training speed (tokens/second)

### Step 3: Generate Text

After training, you can generate text using the trained model:

```bash
python scripts/craft generate text \
    --model models/shakespeare/final_model.pt \
    --prompt "To be or not to be" \
    --max-length 200 \
    --temperature 0.8
```

## Creating a Custom Configuration

Craft uses YAML configuration files to define models, datasets, and training parameters.

### Model Configuration

Create a model configuration in `conf/models/`:

```yaml
# conf/models/my_transformer.yaml
model_type: language
architecture: transformer
vocab_size: 128
d_model: 256
n_head: 4
d_hid: 512
n_layers: 4
dropout: 0.1
```

### Data Configuration

Create a data configuration in `conf/data/`:

```yaml
# conf/data/my_dataset.yaml
type: text
format: character
data_path: data/raw/my_text.txt
batch_size: 64
sequence_length: 128
train_split: 0.9
```

### Training Configuration

Create a training configuration in `conf/training/`:

```yaml
# conf/training/my_training.yaml
epochs: 20
learning_rate: 0.001
optimizer:
  name: adamw
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.1
scheduler:
  name: cosine
  warmup_steps: 100
max_grad_norm: 1.0
mixed_precision: true
```

### Experiment Configuration

Create an experiment configuration in `conf/experiments/`:

```yaml
# conf/experiments/my_experiment.yaml
name: my_transformer_experiment
model_config: conf/models/my_transformer.yaml
data_config: conf/data/my_dataset.yaml
training_config: conf/training/my_training.yaml
paths:
  output_dir: models/my_experiment
  logs_dir: logs/my_experiment
system:
  seed: 42
  device: auto
```

Run your custom experiment:

```bash
python scripts/craft train language --config conf/experiments/my_experiment.yaml
```

## Next Steps

Now that you're familiar with the basics of Craft, you might want to:

- Explore the [architecture documentation](../architecture.md) to understand how Craft works.
- Learn how to [extend Craft](../extending.md) with custom models, datasets, and trainers.
- Contribute to the project by following the contribution guidelines.

Happy modeling! 