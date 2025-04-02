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
*   **`callbacks.py`:** Defines the `Callback` base class and provides a `CallbackList` container, along with implementations of common callbacks (e.g., Logging, EarlyStopping - *to be added*).
*   **`checkpointing.py`:** Contains the `CheckpointManager` class for saving and loading model states, optimizer states, and training progress.
*   **`evaluation.py`:** Contains the `Evaluator` class for evaluating model performance on validation or test datasets.
*   **`progress.py`:** Provides the `ProgressTracker` class for tracking and displaying training progress (e.g., loss, throughput, ETA).
*   **`generation.py`:** Contains the `TextGenerator` class, a wrapper for models with a built-in `.generate()` method (e.g., from Hugging Face).
*   **`sampling.py`:** Contains standalone functions (`generate_text_sampling`, `sample_text`) for text generation using various sampling strategies (temperature, top-k, top-p).
*   **`beam_search.py`:** Contains the standalone `beam_search_generate` function.
*   **`batch_generation.py`:** Contains the standalone `batch_generate` function for parallel generation from multiple prompts.
*   **`amp.py`:** Contains helpers related to automatic mixed precision (AMP) training.
*   **`checkpoint_utils.py`:** Utilities for applying gradient checkpointing to models.
*   **`optimizations.py`:** Contains specific optimization utilities (e.g., mixed precision, torch.compile, activation checkpointing) which are used and tested.

## Guidelines

*   **Trainer Interaction:** Most interactions with this subsystem should happen via the main `scripts/train.py` script, which configures and runs the `Trainer`.
*   **Configuration:** Use Hydra configuration files (`conf/`) to set parameters for the trainer, optimizer, scheduler, and callbacks.
*   **Factories:** Use the `create_optimizer` and `create_scheduler` functions to instantiate these components based on the config.
*   **Generation:** 
    *   For models with a `.generate()` method (like Hugging Face models), use the `TextGenerator` class (configured and potentially used within the `Trainer` or standalone).
    *   For custom models or direct control over sampling/beam search, use the standalone functions imported from `sampling`, `beam_search`, or `batch_generation`.
*   **Callbacks:** To add custom logic during training (e.g., custom logging, metric calculation, model manipulation), create a new class inheriting from `Callback` in `callbacks.py` and implement the desired `on_...` methods (e.g., `on_epoch_end`, `on_step_begin`). Register the new callback in your Hydra config under the `callbacks` list.
*   **Adding Schedulers/Optimizers:** To support new types, update the respective factory functions in `optimizers.py` or `schedulers.py` with the necessary logic and validation.