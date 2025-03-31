# Project Conventions

This document outlines specific conventions, standards, and decisions made for *this particular project*, complementing the general Flow system guidelines.

## Directory Structure for Experiments

After discussion and evaluation of alternatives, we have decided on the following **Hybrid Directory Structure** for managing experiment artifacts:

*   **TensorBoard Logs:**
    *   Location: `outputs/tensorboard/<experiment_id>/`
    *   Purpose: Centralized logging per conceptual experiment (`experiment_id` defined in config) to allow for continuous visualization of resumed training runs in TensorBoard.
    *   Implementation: The `TensorBoardLogger` callback is configured with this path. The `train_runner.py` script ensures that when resuming from a checkpoint, the logger uses the `experiment_id` loaded from the checkpoint's saved config.

*   **Hydra Run Outputs (Checkpoints, Configs, Run Logs):**
    *   Location: `outputs/hydra/<YYYY-MM-DD>/<HH-MM-SS>/`
    *   Purpose: Each individual execution (initial run, resume, variation) gets its own timestamped directory managed by Hydra. This preserves the exact configuration (`.hydra/`), overrides, and run-specific logs (`*.log`) for that execution, aiding reproducibility and debugging.
    *   Implementation: Configured via `hydra.run.dir` in `conf/config.yaml`.

*   **Checkpoints:**
    *   Location: `outputs/hydra/<YYYY-MM-DD>/<HH-MM-SS>/checkpoints/` (Default subdir within the Hydra run dir)
    *   Purpose: Checkpoints are stored with the specific run that generated them.

### Reasoning

This Hybrid approach was chosen because:

1.  **Solves TensorBoard Resumption:** It directly addresses the primary goal of visualizing interrupted training sessions as continuous runs in TensorBoard.
2.  **Preserves Reproducibility:** It retains Hydra's automatic tracking of run-specific configurations and logs, which is crucial for understanding exactly how each checkpoint was produced.
3.  **Balances Trade-offs:** It avoids the higher implementation complexity of a fully collocated structure while being significantly better than the default Hydra setup for experiment visualization.
4.  **Flexibility:** It keeps the door open for integrating more advanced experiment tracking tools (MLflow, W&B) later if needed.

The main trade-off is that artifacts are not fully collocated, requiring separate management of the `outputs/tensorboard/` directory and the `outputs/hydra/` directories when deleting or archiving a complete experiment. 