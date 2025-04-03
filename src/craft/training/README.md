# Training Subsystem

## Purpose

This directory contains the core components for training models within the `craft` framework. The training subsystem is designed to be modular and flexible, allowing for easy configuration and extension of the training process through configuration files (Hydra) and a callback system.

## Structure

The subsystem is composed of the following key modules:

*   **`__init__.py`:** Makes key components importable from `craft.training`.
*   **`trainer.py`:** Contains the `Trainer` class, which orchestrates the overall training process, including epoch management, evaluation, checkpointing, and callbacks.
*   **`training_loop.py`:** Contains the `TrainingLoop` class, encapsulating the logic for a single training epoch: batch iteration, forward/backward passes, gradient accumulation, and optimizer steps.
*   **`optimizers.py`:** Provides the `create_optimizer` factory function to instantiate optimizers (e.g., AdamW) based on configuration.
*   **`schedulers.py`:** Provides the `create_scheduler` factory function to instantiate learning rate schedulers (e.g., CosineAnnealingLR) based on configuration.
*   **`callbacks.py`:** Implements the callback system and various callbacks like `ReduceLROnPlateauOrInstability`, `EarlyStopping`, `SampleGenerationCallback`, and `TensorBoardLogger`.
*   **`checkpointing.py`:** Manages saving and loading model/optimizer/scheduler states.

## Guidelines

*(Specific guidelines for using or extending the training subsystem can be added here.)*