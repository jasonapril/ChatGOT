# Configuration (`conf/`)

This directory holds configuration files for the Craft framework, primarily using YAML format managed by [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/).

## Purpose

- **Define Default Parameters:** Set default values for model architectures, data processing, training hyperparameters, logging, etc.
- **Enable Experimentation:** Easily define and manage different experiment configurations by overriding defaults.
- **Component Instantiation:** Use Hydra's `_target_` key to specify the Python classes to be instantiated for various components (models, datasets, optimizers, etc.).
- **Configuration Validation:** Configurations defined here (or overridden via CLI) are validated at runtime against Pydantic schemas defined in `src/craft/config/schemas.py`. This ensures type safety and consistency.

## Structure

Configurations are organized into logical groups:

```
conf/
├── config.yaml             # Main entry point, defines defaults group
├── data/                   # Data loading and processing configs
│   ├── my_dataset.yaml
│   └── ...
├── model/                  # Model architecture configs
│   ├── transformer_small.yaml
│   └── ...
├── training/               # Training loop, optimizer, scheduler configs
│   ├── default.yaml
│   └── ...
├── callbacks/              # Callback configurations
│   └── ...
├── logging/                # Logging setup
│   └── default.yaml
└── experiment/             # Complete experiment configurations (optional)
    ├── experiment_001.yaml
    └── ...
```

## Composition Logic (Important!)

- The main `config.yaml` **does not** load default configurations directly. It requires an `experiment` to be chosen.
- The selected **`experiment/*.yaml` file is responsible for composing its full configuration**. It uses a `defaults` list to specify which configuration file to load from each required group (e.g., `model`, `data`, `training`, `optimizer`).
- The standard way to specify these defaults within an experiment file is: `- group_name: config_name` (e.g., `- optimizer: adamw`, `- training: default`). Using a leading slash (`- /group_name: config_name`) can prevent inheritance from parent configurations if needed.
- This ensures experiments are self-contained and explicitly declare their components.

## Instantiation via Hydra

A key aspect of this configuration system is the use of `hydra.utils.instantiate`. Many components of the training pipeline are automatically created based on their configuration files:

-   **How it works:** Configuration files for components like models, optimizers, schedulers, datasets, tokenizers, and callbacks should include a `_target_` key specifying the full Python path to the class or function to be instantiated (e.g., `_target_: torch.optim.AdamW`).
-   **Where it happens:** Instantiation primarily occurs within the `Trainer` class (`src/craft/training/trainer.py`) during its initialization. The `Trainer` receives the composed `experiment_config` and uses `hydra.utils.instantiate` to create the necessary objects based on the `_target_` and other parameters defined in the configuration.
-   **Validation:** The parameters specified in the YAML files for a given component should match the arguments expected by the `__init__` method of the `_target_` class/function. Furthermore, higher-level configurations (like `training`) are often validated against Pydantic models defined in `src/craft/config/schemas.py`.
-   **Benefits:** This significantly reduces boilerplate code for object creation and makes swapping components (e.g., trying a different optimizer or dataset) as simple as changing a line in a YAML file or overriding it via the command line.
-   **Requirement:** For a component to be instantiated, its corresponding Python class/function must exist and be importable, and its configuration must correctly define the `_target_` and any required parameters.

## Guidelines

- Define reusable configuration options within the specific group directories (e.g., `conf/optimizer/adamw.yaml`, `conf/model/transformer_small.yaml`).
- Use `experiment/` files to combine configurations from different groups for specific runs.
- Leverage Hydra's composition (`defaults` list) and command-line override capabilities.
- Use descriptive filenames within groups (e.g., `adamw.yaml`). Use `default.yaml` where a single standard configuration is typically used for a group (e.g., `training/default.yaml`).