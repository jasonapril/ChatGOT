# Tests (`tests/`)

## Purpose

This directory contains all automated tests for the source code located in `src/`. This includes unit tests, integration tests, and potentially end-to-end tests.

## Structure

The structure of `tests/` should mirror the structure of `src/` to make it easy to find tests corresponding to specific modules.
-   `tests/`
    -   `unit/`: Tests for individual components in isolation. (Optional subdivision)
        - `test_utils.py`
        - `models/test_model_a.py`
    -   `integration/`: Tests for interactions between components. (Optional subdivision)
    -   `conftest.py`: Fixtures and configuration for `pytest`.

## Guidelines

- Use `pytest` as the testing framework.
- Test filenames should start with `test_` (e.g., `test_module.py`).
- Test function names should start with `test_` (e.g., `test_functionality`).
- Write clear, focused tests.
- Aim for high test coverage of the `src/` codebase.
- Ensure tests can be run easily (e.g., via a `pytest` command from the root).