# Training (`src/craft/training/`)

This package contains the core logic for training and evaluating models.

- `trainer.py`: Defines the main `Trainer` class, which orchestrates the entire training process, including setup, the training loop, evaluation, checkpointing, and callbacks.
- `training_loop.py`: Implements the logic for a single training epoch, including forward/backward passes, optimization steps, gradient accumulation, and metric calculation.
- `evaluation.py`: Contains the `Evaluator` class for running model evaluation on validation/test datasets.
- `checkpointing.py`: Handles saving and loading of training state (model weights, optimizer state, etc.) via the `CheckpointManager`.
- `callbacks/`: Defines the `Callback` interface and implementations for various actions during training (e.g., logging, learning rate scheduling, sample generation, early stopping).
- `generation.py`: Provides the `TextGenerator` class for generating text sequences from a trained model.
- `progress.py`: Utilities for tracking and displaying training progress.
- `sampling.py`: Lower-level sampling functions (e.g., top-k, top-p) used during generation.
- Other files (`amp.py`, `optimizations.py`, etc.): Support utilities for specific training features.

Training pipelines are configured via `conf/training/`, `conf/optimizer/`, `conf/scheduler/`, `conf/callbacks/`, etc., and managed by the `Trainer` class.

## Purpose

This directory contains the core components for training models within the `craft` framework. The training subsystem is designed to be modular and flexible, allowing for easy configuration and extension of the training process through configuration files (Hydra) and a callback system.

## Structure

The subsystem is composed of the following key modules:

*   **`__init__.py`:** Makes key components importable from `craft.training`.
*   **`trainer.py`:** Contains the `Trainer` class, which orchestrates the overall training process, including component instantiation (via Hydra), epoch management, evaluation, checkpointing, and callbacks.
*   **`training_loop.py`:** Contains the `TrainingLoop` class, encapsulating the logic for a single training epoch: batch iteration, forward/backward passes, gradient accumulation, and optimizer steps.
*   **`callbacks/`:** Subdirectory containing callback implementations like `SampleGenerationCallback` and `TensorBoardLogger`.
*   **`checkpointing.py`:** Manages saving and loading model/optimizer/scheduler states.
*   **`evaluation.py`:** Contains logic for evaluating model performance on validation/test sets.
*   **`generation.py`:** Defines the `TextGenerator` class used for creating text samples (e.g., by callbacks or the `Trainer.generate_text` method).
*   **Utilities:** Other files like `amp.py`, `metrics.py`, `progress.py`, etc., provide supporting functionality for the training process.

## Guidelines

*(Specific guidelines for using or extending the training subsystem can be added here.)*