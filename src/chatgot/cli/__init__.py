"""Command-line interface for the ChatGoT project."""
import sys
from .main import main

# Define entry point for use in pyproject.toml
def entry_point():
    """Entry point for setuptools."""
    return main() 