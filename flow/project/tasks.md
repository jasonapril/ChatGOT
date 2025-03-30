# Tasks

*Part of the Flow System. See also: [Guidelines](../system/guidelines.md), [Improvements](../planning/improvements.md).*

This file serves as the working memory for all active tasks in the project. It's designed to provide at-a-glance visibility into what's currently being worked on, what's coming up next, and what has recently been completed.

## Active Tasks

- **Testing Strategy:** 🟡 ⏳ Throughout the refactoring process, ensure comprehensive test coverage is maintained and improved. This includes unit tests for individual components, integration tests for interactions between components, and end-to-end (feature) tests for verifying complete workflows (e.g., training, generation). Add specific testing sub-tasks to relevant refactoring items.

- #### Experiment: Train Larger Model with Longer Context (Character-Level)
  - **Name**: `got_char_1M_ctx256_bs64` # Assigned Experiment Name
  - **Goal**: Evaluate training performance (throughput) and model quality (loss, generation) with increased model capacity and context length on consumer hardware.
  - **Configuration**:
    - Model: ~1M parameters (`d_model=128, n_layers=6, n_head=8, d_hid=512`)
    - Context Length: 256 (`block_size=256`)
    - Tokenization: Character-level
    - Training: 5 epochs, `batch_size=64`, `gradient_accumulation_steps=1`
  - **Status**: ⏸️ Paused (Ready to resume from latest checkpoint)
  - **Latest Checkpoint**: `outputs/2025-03-29/19-28-11/checkpoints/checkpoint_step_8570.pt`
  - **Context**: Reverted block_size from 1024 to 256 due to performance issues. This run uses batch_size 64 to evaluate performance against previous baselines after code refactoring.
  - **Metrics**: Track Tokens/sec, loss curves (step & epoch), qualitative generation results.

### 🔴 High Priority

*These tasks represent major foundational refactoring.* 

- [✅] **Complete Model Architecture Refactoring** (Est. Effort: High) - Standardize model interfaces, config handling (Pydantic), factory, registration.
    - [x] Define model abstraction hierarchy (`Model` -> `GenerativeModel` -> `LanguageModel`).
    - [x] Refactor existing models (`Transformer`, `GPTDecoder`) to fit the new structure.
    - [x] Integrate `generate` method into `GenerativeModel`.
    - [x] Update model loading/saving to use `state_dict` and config.
    - [x] Ensure consistent naming and structure in `src/models/`.
    - [x] Add unit tests for base classes, factory, and `TransformerModel` (including `generate`).
- [ ] **Standardize Data Pipeline** (Est. Effort: High) - Create unified Dataset/DataLoader logic.
    - [x] Define base class `BaseDataset` (`src/data/base.py`).
    - [x] Refactor `CharDataset` to inherit from `BaseDataset` and use config (`src/data/dataset.py`).
    - [x] Refactor factory `create_dataset_from_config` for Hydra/fallback (`src/data/base.py`).
    - [x] Verify default `collate_fn` works for `CharDataset`.
    - [x] Add unit tests for `BaseDataset`, `CharDataset`, and factory.
    - [ ] Define standard tokenizer configuration and loading strategy (Documented in `src/data/base.py`).
    - [ ] Implement unified preprocessing steps (if needed beyond `collate_fn`).
    - [ ] Implement custom `collate_fn` for padding/tokenization (when needed).
- [ ] **Refactor Training Loop** (Est. Effort: Medium) - Decouple components, improve clarity.

### 🟠 Medium Priority

*These tasks improve existing components or address known gaps.*

- #### Improve Configuration Handling
  - **Sub-tasks**:
    - ⏳ Improve Configuration Validation (Pydantic/JSONSchema) (`src/config/config_manager.py`)
      - *Prerequisite for: Refactor Dataset Loading*
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Fix Config Defaults (Paths in `conf/*.yaml`) ✅

- #### Refactor Core Training Logic
  - **Sub-tasks**:
    - ⏳ Review Scheduler T_max Calculation (`src/training/base.py::_create_scheduler`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Integrate Optimizations (`src/training/optimizations.py` -> `Trainer`)
      - *See also: Deprecate Old Training Scripts Task*
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Deprecate/Remove Old Training Scripts (`train_runner.py`, `training_loop.py`, etc.)
      - *Depends on: Integrate Optimizations*
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Integrate Proven Training Logic (`scripts/train_with_samples.py` -> `Trainer`) ✅

- #### Refactor Specific Components
  - **Sub-tasks**:
    - ⏳ Refactor Data Handling Details (`CharDataset`, Vocab, Splitting)
      - *Contributes to: Standardize Data Pipeline Goal*
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Remove Unused PositionalEncoding (`src/models/transformer.py`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Clarify/Consolidate Model Implementations & Generation (`gpt_decoder.py`, `transformer.py`, `generation.py`)
      - *Contributes to: Complete Model Architecture Refactoring Goal*
      - 🔍 Identified during code review 2025-03-28

- #### Improve Testing & CI
  - **Sub-tasks**:
    - ⏳ Implement Tests for Remaining Callbacks (`SampleGenerationCallback`, `TensorBoardLogger`)
      - **Status**: ⏸️ Paused (See details below)
      - 🔍 Identified as missing during review 2025-03-29
    - ⏳ Refactor Test Structure and Runner (`tests/run_all_tests.py`, directory org)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Fix Test Import Handling (Remove `try-except-mock`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Improve Generation Test Coverage (`GenerationTests`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Refactor Data Test Configuration (`DataTests`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Fix Integration Tests (`tests/integration_tests.py`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Set up CI/CD pipeline (GitHub Actions)
      - 🔍 Planned

- #### Documentation & Dependencies
  - **Sub-tasks**:
    - ⏳ Update Architecture Documentation ✅
    - ⏳ Investigate Model "Memory Architecture" (`docs/model.md` investigation)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Update Data Pipeline Documentation (`docs/data_pipeline.md`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Correct Config Import in README ✅
    - ⏳ Update Model Documentation (Parameter Count Difference) (`docs/model.md` update)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Verify `transformers` Dependency (Scheduler Usage)
      - 🔍 Identified during code review 2025-03-28

- ### 🟡 Implement Tests for Remaining Callbacks (Details)
  - **Status**: ⏸️ Paused
  - **Description**: Ensure comprehensive test coverage for all trainer callbacks (`SampleGenerationCallback`, `TensorBoardLogger`).
  - **Context**: 
    - Initially added `TestTensorBoardLogger`.
    - Encountered persistent `TypeError` issues with `@patch` and method signatures, particularly for `TestSampleGenerationCallback`.
    - Current blockers: 
      - `TestSampleGenerationCallback`: `TypeError: setUp() missing 1 required positional argument: 'mock_logging'` despite code appearing correct.
      - `TestTensorBoardLogger`: AssertionErrors related to mock calls not being detected.
    - Paused due to persistent test execution errors blocking progress.

## Refactoring

- [x] Refactor vocabulary handling:
    - [x] Create `preprocess.py` script to calculate and save vocab/mappings from raw data.
    - [x] Update `CharDataset` to load precomputed vocab instead of calculating it.
    - [x] Update model config (`conf/model/transformer.yaml`) with the static vocab size.
    - [x] Update data config (`conf/data/char.yaml`) and `DataManager` if needed to pass vocab path.
    - [x] Remove dynamic vocab update logic from `train_runner.py` and `generate_text.py`.

## Upcoming Tasks

- ### 🟠 Refactor Data Pipeline for Tokenizer Flexibility
  - **Goal**: Modify the data pipeline to support different tokenizers (character, subword, etc.) configured via Hydra, following the strategy outlined in `src/data/base.py`.
  - **Steps**:
    1. Define/Re-introduce standalone `Tokenizer` classes (e.g., `CharacterTokenizer`) in `src/data/tokenizer.py`.
    2. Define corresponding Hydra configs (e.g., `conf/tokenizer/char.yaml`).
    3. Update `conf/config.yaml` to include `tokenizer` defaults/placeholders.
    4. Modify `Dataset` classes (e.g., `CharDataset`) to return raw text instead of tokenizing internally.
    5. Implement custom `collate_fn` functions (e.g., `character_collate_fn` in `src/data/collation.py`) that use the instantiated tokenizer.
    6. Modify `prepare_dataloaders_from_config` to accept the tokenizer, determine the correct `collate_fn`, and pass it to `DataLoader`.
    7. Modify `train_runner.py` to instantiate the `tokenizer` via Hydra and pass it to `prepare_dataloaders_from_config` and `Trainer`.
    8. Modify `Trainer` to accept and use the `tokenizer` object for sampling, removing `vocab_path`.
  - **Status**: ⏳ To Do (Postponed from previous attempt)

- ### 🟡 Review `benchmarking\\benchmarks` vs `outputs\\benchmarks`

- ### 🔴 Define Goals
  - We should define the precise goals that we wish to attain with this project, and work backwards from there to ensure that the project's implementations are ideal for fulfilling those goals.
  - Develop AI models that work on low-spec devices (performance critical)
  - Experiment with cutting edge AI architectures (research)

- ### 🟠 Review Dependencies
  - Which libraries/frameworks/packages/etc. does Craft depend on? Are they all justified and ideal? What are the pros and cons compared to alternatives? When does it make sense to use custom libraries/frameworks/packages/etc.?

- ### 🟡 Improve CLI Generation Tokenization
  - Refactor `src/cli/run.py::generate_text` to load tokenizer based on model checkpoint
  - Remove `TODO` placeholder logic
  - Ensure compatibility with different model types/tokenizers
  - 🔍 Identified during code review 2025-03-28

## Completed Tasks (Recent)

- ### 🟡 Implement Basic Unit Tests ✅ 2025-03-28
- ### 🟡 Implement Folder Structure ✅ 2025-03-28
- ### 🔴 Update Architecture Documentation ✅ 2025-03-28
- ### 🟡 Fix Config Defaults ✅ 2025-03-28
- ### 🔴 Integrate Proven Training Logic ✅ 2025-03-28
- ### 🟡 Complete Config Refactor (`configs` -> `conf`) ✅ 2025-03-28