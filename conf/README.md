# Configuration (`conf/`)

## Purpose

This directory stores all configuration files for the project, primarily using YAML format managed by [Hydra](https://hydra.cc/) / [OmegaConf](https://omegaconf.readthedocs.io/). It allows for flexible and reproducible experiment setup. This structure follows standard Hydra conventions, promoting modularity, reusability, and reproducibility in configurations.

## Structure

Configurations are organized into groups based on their function:
-   `conf/`
    -   `config.yaml`: Main configuration file, often used as the entry point for Hydra. Defines defaults and structure.
    -   `experiment/`: Complete experiment configurations. These files compose settings from other groups.
    -   `data/`: Configurations related to datasets, preprocessing, and data loading.
    -   `model/`: Configurations defining model architectures and hyperparameters.
    -   `training/`: Configurations for the main training loop (epochs, batch size, validation, etc.).
    -   `optimizer/`: Configurations for different optimizers (e.g., AdamW, SGD) and their parameters.
    -   `scheduler/`: Configurations for learning rate schedulers.
    -   `callbacks/`: Configurations for training callbacks (e.g., checkpointing, early stopping).
    -   `tokenizer/`: Configurations related to tokenizers (if applicable and managed separately).
    -   `README.md`: This file.

## Guidelines

- Define reusable configuration defaults or options within the specific group directories (e.g., `conf/optimizer/adamw.yaml`).
- Use `experiment/` files to combine configurations from different groups for specific runs.
- Leverage Hydra's composition (`defaults` list) and override capabilities.
- The `config.yaml` often sets the default choices for each group.