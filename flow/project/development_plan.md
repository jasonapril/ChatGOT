`# Development Plan (Created: 2025-04-01)

# Initial Development Plan for Craft Framework

## Goal

Establish the core functionality of the Craft framework required to train baseline language models (character-level and subword-level) of varying sizes (e.g., 1M, 10M params) on initial datasets (Game of Thrones, Wikipedia subset), enabling logging, checkpointing, sample generation, and basic evaluation.

## Development Phases (Iterative Approach)

We will build and test features incrementally, focusing first on the smaller Game of Thrones dataset and smaller models to ensure the end-to-end pipeline works before scaling to Wikipedia.

### Phase 1: Core Infrastructure & Data Pipeline (GoT Focus)

1.  **Configuration Handling (`src/craft/config`, `conf/`)**
    *   Implement robust loading and validation of Hydra configurations (`.yaml` files).
    *   Define base configuration structures (`conf/config.yaml`) and initial configs for data (`conf/data/got_char.yaml`), models (`conf/model/transformer_small.yaml`), training (`conf/training/default.yaml`), optimizers (`conf/optimizer/adamw.yaml`), etc.
    *   Ensure configurations are accessible within the framework code via `src/craft/config`.

2.  **Data Processing & Loading (`src/craft/data`, `scripts/prepare_data.py`, `data/`)**
    *   Implement Character-level processing for GoT (logic within `src/craft/data/processors.py` or called by `scripts/prepare_data.py`).
        *   Input: `data/raw/got.txt` (or similar path)
        *   Output: Tokenized `train/val/test` splits in `data/processed/got/char_level/` (e.g., `.pkl` files containing token IDs, vocab info).
    *   Implement `Dataset` and `DataLoader` logic (`src/craft/data/datasets.py`, `src/craft/data/dataloaders.py`) capable of loading the processed character-level data based on configuration.
    *   Ensure `scripts/prepare_data.py` can be driven by configurations in `conf/data/`.

3.  **Basic Model Definition (`src/craft/models`)**
    *   Implement a basic Transformer architecture (`src/craft/models/transformer.py`).
    *   Parameterize the model (layers, heads, dimensions) so it can be configured via `conf/model/*.yaml`.
    *   Implement model creation based on config (`src/craft/models/factory.py` or similar, using `create_model_from_config`).
    *   Target initial small model sizes (e.g., ~1M params).

4.  **Core Training Loop (`src/craft/training`, `scripts/train.py`)**
    *   Implement a basic `Trainer` class (`src/craft/training/trainer.py` or `base.py`).
    *   Include essential steps: batch iteration, forward pass, loss calculation (CrossEntropy), backward pass, optimizer step.
    *   Integrate configuration for training parameters (learning rate, batch size, epochs) from `conf/training/` and `conf/optimizer/`.
    *   Ensure `scripts/train.py` can initiate training using an experiment config (`conf/experiment/got_char_small.yaml`).

5.  **Basic Testing (`tests/`)**
    *   Add unit tests for config loading, data loading/processing, model instantiation, and core training components (forward/loss).
    *   Ensure tests run locally (`pytest tests/`) and via the CI workflow.

### Phase 2: Training Essentials & Evaluation

1.  **Logging (`src/craft/training`, `src/craft/utils/logging.py`, `outputs/`)**
    *   Integrate basic console logging effectively throughout the training process.
    *   Integrate TensorBoard logging (`outputs/tensorboard/`) for key metrics (loss, learning rate, validation metrics). Configure via `conf/training/` or callbacks (`conf/callbacks/`).

2.  **Checkpointing (`src/craft/training`, `outputs/`)**
    *   Implement model checkpoint saving (e.g., based on epoch or validation performance) to the Hydra run directory (`outputs/hydra/.../checkpoints/`). Include optimizer state. Configure via `conf/callbacks/`.
    *   Implement logic to resume training from a specified checkpoint.

3.  **Basic Evaluation (`src/craft/training`, `scripts/evaluate.py`)**
    *   Implement calculation of validation loss/perplexity within the `Trainer`'s validation loop.
    *   Develop `scripts/evaluate.py` to load a checkpoint and calculate metrics (loss, perplexity) on a specified dataset split (e.g., test set).

4.  **Sample Generation (`src/craft/models`, `src/craft/utils/generation.py`, `scripts/generate.py`)**
    *   Implement a `generate` method within the base model class or specific models (`src/craft/models/transformer.py`).
    *   Develop `scripts/generate.py` to load a checkpoint and generate text samples based on a prompt, using configured generation parameters (max length, temperature).

### Phase 3: Scaling & Subword Tokenization (Wiki Prep)

1.  **Subword Tokenization (`src/craft/data`, `conf/tokenizer`, `scripts/prepare_data.py`)**
    *   Integrate/implement subword tokenizer training (e.g., BPE using Hugging Face `tokenizers` library).
    *   Update data processing (`scripts/prepare_data.py`, `src/craft/data/processors.py`) to handle subword tokenization for GoT (`data/processed/got/subword_level/`).
    *   Update `Dataset/DataLoader` (`src/craft/data/`) to handle subword tokenized data.
    *   Train and evaluate GoT models using subword tokens, comparing with character-level results.

2.  **Wikipedia Preprocessing (`scripts/wiki_preprocess.py`)**
    *   Develop robust script(s) to parse, clean, and extract text from the Wikipedia XML dump. Consider using existing libraries/tools (`datasets`, WikiExtractor).
    *   Process an initial subset (e.g., 1GB) and save to `data/raw/wiki/subset_1gb.txt` (or similar).
    *   Train a subword tokenizer on the Wiki subset (`data/processed/wiki/subset_1gb/tokenizer/`).
    *   Process the Wiki subset using the trained tokenizer (`data/processed/wiki/subset_1gb/`).

3.  **Performance Monitoring & Optimization (`src/craft/performance`)**
    *   Integrate the `performance` module (throughput monitor, instrumentation) into the `Trainer`.
    *   Log performance metrics (tokens/sec, memory usage) to console and TensorBoard.
    *   Implement/test gradient accumulation and mixed-precision training options (configure via `conf/training/`).

4.  **Scaling Model Size (`conf/model`, `src/craft/models`)**
    *   Define configurations for larger models (10M, 100M params).
    *   Train and evaluate these larger models, first on GoT, then on the Wiki subset. Analyze scaling behavior (performance vs. params vs. data).

### Phase 4: Further Scaling & Refinement (Future)

*   Process Full Wikipedia dataset.
*   Explore larger models (1B+ params, potentially requiring distributed training).
*   Implement more advanced evaluation metrics or generation techniques.
*   Refactor and optimize framework based on findings.
*   Explore fine-tuning, RL, tool use as separate future phases.

## Testing Strategy (Moved from tasks.md)

🟡 ⏳ Throughout the refactoring process, ensure comprehensive test coverage is maintained and improved. This includes unit tests for individual components, integration tests for interactions between components, and end-to-end (feature) tests for verifying complete workflows (e.g., training, generation). Add specific testing sub-tasks to relevant refactoring items. 