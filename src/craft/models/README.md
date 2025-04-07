# Models (`src/craft/models/`)

This directory contains neural network model implementations and base classes.

- `base.py`: Defines base classes for models (e.g., `BaseModel`, `LanguageModel`, `GenerativeModel`), establishing common interfaces and functionality.
- Specific model architecture files (e.g., `transformer.py`, `simple_rnn.py`): Implement concrete model architectures, often inheriting from the base classes.

Models are typically configured via the `conf/model/` directory and instantiated by Hydra based on the `_target_` specified in the configuration, pointing to a class within this package. 