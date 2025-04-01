# Craft Tests

This directory contains tests for the Craft project.

## Structure

Tests are organized mirroring the `src/craft` directory structure:

*   `tests/data/`: Tests for data loading (`PickledDataset`, `create_data_loaders_from_config`), processing (`process_char_level_data`), and related utilities.
*   `tests/models/`: Tests for model architectures (e.g., `TransformerModel`) and base classes.
*   `tests/training/`: Tests for training loop components (`Trainer`, `TrainingLoop`), generation (`generate_text`), etc.
*   `tests/integration/`: End-to-end integration tests (Planned).
*   `tests/`: Contains shared configuration (`conftest.py`).

## Running Tests

Tests are written using `pytest` (and `unittest` for some older files).

To run all tests from the root directory:

```bash
pytest
```

To run tests in a specific file or directory:

```bash
pytest tests/data/test_datasets.py
pytest tests/models/
```

Coverage information is automatically generated if `pytest-cov` is installed.