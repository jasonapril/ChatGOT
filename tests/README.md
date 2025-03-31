# Tests (`tests/`)

This directory contains automated tests for the Craft project.

## Structure:

*   `unit/`: Contains unit tests that focus on testing individual components (functions, classes) in isolation. Mocking is often used here to isolate the unit under test.
*   `integration/`: (If present) Contains integration tests that verify the interaction between multiple components.
*   Other test files (e.g., `test_models.py`, `test_data.py`): May contain tests that span multiple units or test specific high-level functionalities.
*   `fixtures/` or `data/`: (If present) May contain test data or fixture files used by the tests.

## Framework:

Tests primarily use the standard Python `unittest` framework. Consider migrating to or integrating with `pytest` for enhanced features if needed.

## Running Tests:

Tests can typically be run using a test runner. From the project root directory:

*   **Using `unittest` discovery:**
    ```bash
    python -m unittest discover tests
    ```
*   **Running a specific file:**
    ```bash
    python -m unittest tests/unit/test_optimizations.py
    ```
*   **If `pytest` is installed:**
    ```bash
    pytest
    ```
    or
    ```bash
    pytest tests/
    ``` 