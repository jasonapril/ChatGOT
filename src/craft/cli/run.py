#!/usr/bin/env python
"""
Command-line interface for Craft.

This module provides a unified CLI for working with different model types
and experiments by composing commands from separate modules.
"""
import logging
import typer
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
import sys
import os

# Temporarily add the project root to the Python path

# Import command modules
from .train_commands import train_app
from .dataset_commands import dataset_app
from .evaluate_commands import evaluate_app
from .generate_commands import generate_app

# Create the main Typer app
app = typer.Typer(
    name="craft",
    help="A framework for developing AI models",
    add_completion=False,
)

# Add subcommands from imported modules
app.add_typer(train_app, name="train", help="Training related commands")
app.add_typer(dataset_app, name="data", help="Data processing and tokenization commands")
app.add_typer(evaluate_app, name="eval", help="Evaluation related commands")
app.add_typer(generate_app, name="generate", help="Text generation related commands")

# Global console (optional, can be shared or each module can have its own)
console = Console()

# Shared logging setup function
def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging with rich handler."""
    # Make sure logging level is valid
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
        
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s", # Rich handler handles formatting
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)] # Configure RichHandler
    )
    # Set level for httpx logger if it exists and is noisy
    # logging.getLogger("httpx").setLevel(logging.WARNING)

# Main app callback for global options like log level
@app.callback()
def run_command(ctx: typer.Context) -> None:
    """
    Top-level callback, can be used for global setup based on context.
    """
    # Consider removing verbose, let log_level control it directly
    # verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output (sets log level to DEBUG)"),
    log_level: str = ctx.params.get("log_level", "INFO")
    # if verbose:
    #     level_to_set = "DEBUG"
    try:
        setup_logging(log_level)
        logging.getLogger(__name__).debug(f"Log level set to {log_level}")
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        # Catch other potential logging setup errors
        console.print(f"[bold red]Error setting up logging:[/bold red] {e}")
        raise typer.Exit(code=1)

def main() -> None:
    """Main entry point for the CLI application."""
    # Potentially add global setup here if needed
    # e.g., setup_logging(level="INFO") if not handled by commands
    app()

if __name__ == "__main__":
    main() # -> None added by adding return type to main 