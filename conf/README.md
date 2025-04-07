# Configuration (`conf/`)

## Purpose

This directory stores default configuration files for the project, primarily using YAML format managed by [Hydra](https://hydra.cc/) / [OmegaConf](https://omegaconf.readthedocs.io/). It allows for flexible and reproducible experiment setup by providing composable defaults.

**Validation:** While these files define default *values*, the *structure* and *types* are enforced by the Pydantic schemas defined in `src/craft/config/schemas.py`. Hydra loads these YAML files, composes them based on the `defaults` list in an experiment config, and the resulting configuration object is typically validated against the corresponding Pydantic schema (e.g., `TrainingConfig`, `AppConfig`) upon instantiation or use.

## Structure

Configurations are organized into groups based on their function:
-   `conf/`
    -   `config.yaml`: **Main configuration entry point**. Defines top-level settings and Hydra behavior. **Crucially, it mandates that an `experiment` configuration must be specified** (e.g., `experiment=my_exp`).
    -   `generate.yaml`: Standalone configuration for the generation script (`scripts/generate.py`).
    -   `experiment/`: Contains complete experiment configurations. These files compose settings from other groups using the `defaults` list (e.g., `- /model: transformer_small`, `- /data: my_dataset`).
    -   `data/`: Configurations related to datasets, tokenizers, and data loading. Should specify a `_target_` for instantiation (e.g., `craft.data.dataset.PickledDataset`).
    -   `model/`: Configurations defining model architectures and hyperparameters. Should specify a `_target_` (e.g., `craft.models.transformer.TransformerModel`).
    -   `training/`: Configurations for the training loop itself (e.g., epochs, batch size, learning rate, intervals). Parameters here correspond to the `TrainingConfig` Pydantic schema.
    -   `optimizer/`: Configurations for different optimizers. Must specify a `_target_` (e.g., `torch.optim.AdamW`) and relevant parameters.
    -   `scheduler/`: Configurations for learning rate schedulers. Must specify a `_target_` (e.g., `torch.optim.lr_scheduler.CosineAnnealingLR`).
    -   `callbacks/`: Configurations defining training callbacks. Each callback needs a `_target_` (e.g., `craft.training.callbacks.TensorBoardLogger`). The top-level file (e.g., `default.yaml`) typically defines a dictionary of callbacks.
    -   `generation/`: Default configurations for text generation parameters (used during training sampling and potentially by `generate.yaml`).
    -   `hydra/`: Internal Hydra configuration (logging, sweep settings, run directories).
    -   `README.md`: This file.

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