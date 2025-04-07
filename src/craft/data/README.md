# Data Handling (`src/craft/data/`)

This package contains modules related to data loading, processing, and tokenization.

- `datasets/`: Contains PyTorch `Dataset` implementations for different data formats (e.g., `pickled_dataset.py` for pre-tokenized data, `text_dataset.py` for handling raw text).
- `tokenizers/`: Defines the base `Tokenizer` interface (`base.py`) and specific tokenizer implementations (e.g., `char.py`, `sentencepiece.py`).
- `utils.py`: Utility functions related to data handling (e.g., creating dataloaders).
- `base.py`: May contain base classes or common functionality for data components.

Configuration for data components (datasets, tokenizers, dataloader parameters) is typically managed in `conf/data/` and instantiated via Hydra. 