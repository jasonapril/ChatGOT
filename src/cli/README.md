# Command-Line Interface (`src/cli/`)

This directory contains the code for the ChatGoT command-line interface (CLI), built using the [Typer](https://typer.tiangolo.com/) library.

## Structure:

*   `run.py`: The main entry point for the CLI (`python -m src.cli.run ...`). It defines the top-level `typer.Typer()` application, handles global options (like `--log-level`), sets up logging, and imports and registers command groups (sub-apps) from other modules in this directory.
*   `train_commands.py`: Defines the `train_app` and includes commands related to model training (e.g., `train language`).
*   `dataset_commands.py`: Defines the `dataset_app` and includes commands related to dataset operations (e.g., `dataset prepare`).
*   `evaluate_commands.py`: Defines the `evaluate_app` and includes commands related to model evaluation (currently a placeholder).
*   `generate_commands.py`: Defines the `generate_app` and includes commands related to text generation (currently a placeholder).

## Purpose:

The CLI provides a user-friendly interface for interacting with the core functionalities of the ChatGoT project, such as preparing data, training models, evaluating results, and generating text. It uses [Hydra](https://hydra.cc/) (initialized within commands like `train language`) for managing complex configurations.

## Adding New Commands:

To add new commands or command groups:

1.  Add the command function decorated with `@some_app.command(...)` to the relevant `*_commands.py` file (or create a new file + Typer app if it's a new logical group).
2.  If a new command file/app was created, import its app instance (e.g., `from .new_commands import new_app`) in `run.py`.
3.  Register the app with the main `app` in `run.py` using `app.add_typer(new_app, name="new_group_name")`. 