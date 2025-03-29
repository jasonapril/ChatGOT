# CLI (Command Line Interface) in Craft

This document explains the Command Line Interface (CLI) used in the Carft project, its design, benefits, and usage examples.

## What is CLI?

A Command Line Interface (CLI) is a text-based interface that allows users to interact with a software application by typing commands in a terminal or console. Unlike a Graphical User Interface (GUI) which uses visual elements like buttons and menus, a CLI relies on text commands.

## CLI Architecture in Craft

The Craft project implements a CLI using the [Typer](https://typer.tiangolo.com/) library, which provides a simple and elegant way to build command-line applications in Python.

The CLI structure is organized as follows:

- **Entry point**: `scripts/craft` - A lightweight script that imports and calls the main application
- **Main CLI module**: `src/cli/run.py` - Contains the main CLI application and command definitions
- **Command groups**: The CLI is organized into subcommands like `train`, `generate`, `evaluate`, etc.
- **Command handlers**: Each command is implemented as a function decorated with Typer decorators

## Benefits of CLI for ML Projects

For a machine learning project like Craft, a CLI offers several advantages:

1. **Reproducibility**: Commands with their parameters can be easily documented and rerun exactly as before
2. **Automation**: CLI commands can be included in scripts or batch files for automated workflows
3. **Remote execution**: Training can be run on remote servers/clusters without a GUI
4. **Parametrization**: Clear way to pass various arguments and configurations
5. **Modularity**: Different functions (train, generate, evaluate) are organized as subcommands
6. **Logging**: Easy integration with logging systems for capturing outputs
7. **Documentation**: Self-documenting with help messages and parameter descriptions

## Available Commands

The Craft CLI includes the following main command groups:

- **train**: Commands for model training
  - `train language`: Train a language model
- **generate**: Commands for text generation
  - `generate text`: Generate text from a trained model
- **evaluate**: Commands for model evaluation
- **experiment**: Commands for running experiments
- **dataset**: Commands for dataset operations

## Usage Examples

### Training a Language Model

```bash
python scripts/craft train language \
  --config conf/language_model.json \
  --output-dir runs/my_experiment \
  --device cuda
```

### Generating Text

```bash
python scripts/craft generate text \
  --model models/my_model.pt \
  --prompt "Once upon a time" \
  --max-length 200 \
  --temperature 0.8
```

### Batch Processing with Scripts

The CLI's command structure makes it easy to create batch processing scripts like the ones in `flow/active/`:

- `generate_samples.py` - Python script for generating multiple samples
- `generate_samples.sh` - Bash script for Unix-like systems
- `generate_samples.ps1` - PowerShell script for Windows

## Extending the CLI

To add new commands to the CLI:

1. Add a new function to the appropriate command group in `src/cli/run.py`
2. Decorate the function with the appropriate Typer decorator
3. Document the function and its parameters

Example:

```python
@train_app.command("new_model")
def train_new_model(
    config_path: str = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
):
    """Train a new model type."""
    # Implementation here
```

## Best Practices

When using the CLI:

1. Always specify required parameters explicitly
2. Use configuration files for complex parameter sets
3. Save the exact command used for reproducibility
4. Use version control for configuration files
5. Redirect output to log files for long-running commands 