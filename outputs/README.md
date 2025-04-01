# Outputs (`outputs/`)

## Purpose

This directory serves as the primary location for all generated artifacts from running experiments and scripts within the Craft project. It is designed to keep generated files separate from the source code and configuration.

## Structure

The recommended structure leverages Hydra's output management and common logging practices:

-   `outputs/`
    -   `hydra/`: Contains the detailed outputs (logs, config backups, checkpoints, etc.) for *each individual execution* (run) managed by Hydra, typically organized by date and time (`YYYY-MM-DD/HH-MM-SS/`).
    -   `tensorboard/`: (Optional) Contains aggregated TensorBoard event logs, typically grouped by experiment name (`<experiment_name>/`).
    -   `README.md`: This file.

## Guidelines

-   **Hydra Management:** Hydra should be configured to save run outputs into `outputs/hydra/`. This is often the default behavior when running scripts from the project root.
-   **TensorBoard Logging:** If used, configure TensorBoard loggers to save into `outputs/tensorboard/<experiment_name>/`.
-   **Reproducibility:** The `outputs/hydra/` structure ensures run-specific artifacts and configurations are stored together.
-   **`.gitignore`:** This entire `outputs/` directory should typically be included in `.gitignore`.
-   **Artifact Access:** Find specific run artifacts in `outputs/hydra/`. Find aggregated TensorBoard logs in `outputs/tensorboard/`.