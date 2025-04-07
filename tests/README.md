# Tests (`tests/`)

This directory contains automated tests for the Craft framework, using the `pytest` framework.

## Structure

Tests are organized mirroring the `src/craft/` structure:

- `tests/config/`: Tests for configuration loading and schema validation.
- `tests/models/`: Tests for specific model implementations and base classes.
- `tests/data/`: Tests for datasets, tokenizers, and data utilities.
- `tests/training/`: Tests for the Trainer, training loop, callbacks, checkpointing, etc.
- `tests/cli/`: Tests for the command-line interface.
- `tests/utils/`: Tests for common utility functions.
- `tests/conftest.py`: Contains shared fixtures used across multiple test files.

## Running Tests

Tests can be run from the project root directory using `pytest`:

```bash
pytest
```

To run specific tests:

```bash
pytest tests/training/trainer/test_trainer_init.py
pytest tests/models/test_transformer.py::TestTransformerModel::test_forward_pass_shape
```

Coverage reports can be generated if `pytest-cov` is installed:

```bash
pytest --cov=src/craft
```