# Testing (`tests/`)

This directory contains all the tests for the Craft framework.

## Structure

Tests are organized mirroring the structure of the `src/craft/` directory:

```
tests/
├── conftest.py             # Global fixtures and test setup (e.g., mocking)
├── data/                   # Tests for data loading, tokenizers, datasets
│   ├── test_datasets.py
│   └── test_tokenizers.py
├── models/                 # Tests for model architectures and base classes
│   └── test_transformer.py
├── training/               # Tests for the Trainer, callbacks, checkpointing
│   ├── test_trainer.py
│   ├── test_callbacks.py
│   └── test_checkpointing.py
├── cli/                    # Tests for the command-line interface
│   └── test_cli_commands.py
├── utils/                  # Tests for utility functions
│   └── test_utils.py
└── integration/            # Integration tests covering multiple components (optional)
    └── test_full_training_run.py 
```

## Running Tests

Tests are typically run using `pytest` from the project root directory:

```bash
pytest
```

Specific tests or test files can be run by providing their path:

```bash
pytest tests/training/test_trainer.py
```

## Key Files

- **`conftest.py`**: Contains shared fixtures (e.g., temporary directories, mock objects, pre-configured components) used across different test files. This promotes code reuse and simplifies test setup.
- **Test Modules (`test_*.py`)**: Each module focuses on testing a specific component or functionality from the corresponding `src/craft/` module.

## Philosophy

- **Unit Tests:** Focus on testing individual functions or classes in isolation, often using mocking to replace dependencies.
- **Integration Tests:** (Optional, often placed in `tests/integration/`) Test the interaction between multiple components (e.g., a short training run).
- **Fixtures:** Leverage pytest fixtures extensively for setup and teardown, making tests cleaner and more maintainable.