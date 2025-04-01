# Source Code (`src/`)

## Purpose

This directory contains the Python source code for the project, structured as an installable package named `craft`.

## Rationale for `src/craft/` Layout ("src layout")

Placing the main package code (`craft`) inside the `src/` directory is a standard Python best practice with several advantages:

- **Clear Installability:** Makes it explicit that `craft` is the installable package.
- **Prevents Accidental Imports:** Avoids issues where Python might import the package directly from the root directory during development, ensuring tests run against the installed version.
- **Clean Namespace:** Clearly separates the package code from other project files (like tests, scripts, documentation).
- **Tooling Compatibility:** Works well with standard Python build and testing tools.

## Structure

- `src/`
  - `craft/`: Contains the actual Python package source code. See `src/craft/README.md` for its internal structure.
  - `README.md`: This file.

## Guidelines

- All core library/framework code belongs inside the `src/craft/` directory.
- Follow PEP 8 style guidelines within the package.
- Ensure code is installable (e.g., via `pip install -e .` using `pyproject.toml`).
- Tests for the code in `src/craft/` should reside in the top-level `tests/` directory, mirroring the package structure.