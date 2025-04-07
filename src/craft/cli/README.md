# Command-Line Interface (`src/craft/cli/`)

This package implements the command-line interface (CLI) for the Craft framework using [Typer](https://typer.tiangolo.com/).

- `run.py`: The main entry point for the CLI, defining the top-level `app` and subcommands.
- Other files (e.g., `train_commands.py`, `generate_commands.py`, `dataset_commands.py`): Define the logic and arguments for specific subcommands (like `train`, `generate text`, `dataset prepare`).

## Usage

The CLI is typically invoked via `python -m craft.cli.run`. Available commands and options can be discovered using:

```bash
python -m craft.cli.run --help
python -m craft.cli.run train --help
python -m craft.cli.run generate --help
# etc.
```

The CLI commands often rely on Hydra for configuration management, allowing users to specify experiment configurations and overrides:

```bash
python -m craft.cli.run train experiment=my_experiment training.num_epochs=5
```

## Structure:

*   `run.py`: The main entry point for the CLI (`python -m src.cli.run ...`). It defines the top-level `typer.Typer()` application, handles global options (like `--log-level`), sets up logging, and imports and registers command groups (sub-apps) from other modules in this directory.
*   `dataset_commands.py`: Defines the `dataset_app` and includes commands related to dataset operations (e.g., `dataset prepare`).
*   `train_commands.py`: Defines the `train_app` and includes commands related to model training (e.g., `train language`).
*   `generate_commands.py`: Defines the `generate_app` and includes commands related to text generation (e.g., `generate text`).
*   `evaluate_commands.py`: Defines the `evaluate_app` and includes commands related to model evaluation (currently a placeholder).

## Purpose:

The CLI provides a user-friendly interface for interacting with the core functionalities of the ChatGoT project, such as preparing data, training models, evaluating results, and generating text. It uses [Hydra](https://hydra.cc/) (initialized within commands like `train language`) for managing complex configurations.

## Adding New Commands:

To add new commands or command groups:

1.  Add the command function decorated with `@some_app.command(...)` to the relevant `*_commands.py` file (or create a new file + Typer app if it's a new logical group).
2.  If a new command file/app was created, import its app instance (e.g., `from .new_commands import new_app`) in `run.py`.
3.  Register the app with the main `app` in `run.py` using `app.add_typer(new_app, name="new_group_name")`. 