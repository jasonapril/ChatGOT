# Benchmarking (`benchmarking/`)

This directory contains a suite for running performance and accuracy benchmarks for the Craft project.

## Structure:

*   `main.py`: The main CLI entry point for the benchmarking suite. Provides commands to `run` benchmarks, `list` available benchmarks, and generate `report`s.
*   `runner.py`: Contains the `BenchmarkRunner` class, which handles the logic for discovering, executing, and collecting results from individual benchmarks.
*   `benchmarks/`: Contains the specific benchmark definitions as Python modules (e.g., `inference_performance.py`, `training_speed.py`, `model_accuracy.py`).
*   `utils/`: Utility functions specific to the benchmarking process (e.g., visualization, helper functions).
*   `logs/`: Default directory for storing logs generated during benchmark runs.
*   `results/`: Default directory for storing benchmark results (likely JSON files).

## Usage:

Benchmarks are run via the `main.py` script using the `run` command. From the project root directory:

*   **List available benchmarks:**
    ```bash
    python -m benchmarking.main list
    ```
*   **Run all benchmarks:**
    ```bash
    python -m benchmarking.main run --model-checkpoint /path/to/your/model.pt
    ```
*   **Run specific benchmarks:**
    ```bash
    python -m benchmarking.main run --benchmarks inference_performance training_speed --model-checkpoint /path/to/your/model.pt
    ```
*   **Generate a report from results:**
    ```bash
    python -m benchmarking.main report --results-file /path/to/results.json
    ```

See `python -m benchmarking.main --help` for more options. 