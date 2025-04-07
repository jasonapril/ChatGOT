# Configuration Schemas (`src/craft/config/`)

This directory defines the data structures (schemas) used for validating the configuration of the Craft framework.

- `schemas.py`: Contains Pydantic models that define the expected types, structures, and validation rules for different configuration sections (e.g., `TrainingConfig`, `ModelConfig`, `DataConfig`).

These Pydantic schemas are used to:

1.  **Validate** configuration values loaded from YAML files (via Hydra) or passed directly.
2.  Provide **type hints** and **autocompletion** during development.
3.  Ensure **consistency** and **clarity** in how experiments are configured.

The actual default configuration *values* are typically defined in the `conf/` directory at the project root, managed by Hydra, and then validated against these schemas during runtime. 