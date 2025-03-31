# Tasks

*Part of the Flow System. See also: [Guidelines](../system/guidelines.md), [Improvements](../planning/improvements.md).*

This file serves as the working memory for all active tasks in the project. It's designed to provide at-a-glance visibility into what's currently being worked on, what's coming up next, and what has recently been completed.

## Active Tasks

- **Test End-to-End Workflow**: 🟡 ⏳ Run minimal training test to verify core loop, logging, config after refactors.
  - *Script*: `python -m src.training.train_runner training=minimal_test`
  - *Status*: Ready to run

- **Implement Consistent TensorBoard Logging**: 🟡 ⏸️ Modify configuration and potentially training script to ensure TensorBoard logs from resumed runs append to the original experiment log directory, identified by a unique `experiment_id`.
  - *Depends on*: [Fix and Utilize TensorBoard Logging](#fix-and-utilize-tensorboard-logging-tensorboardlogger) (tests passing)
  - *Context*: Needed to correctly visualize loss curves across interrupted training sessions. (Paused pending basic workflow test)

- **Testing Strategy:** 🟡 ⏳ Throughout the refactoring process, ensure comprehensive test coverage is maintained and improved. This includes unit tests for individual components, integration tests for interactions between components, and end-to-end (feature) tests for verifying complete workflows (e.g., training, generation). Add specific testing sub-tasks to relevant refactoring items.

- #### Experiment: Train Larger Model with Longer Context (Character-Level)
  - **Name**: `got_char_1M_ctx256_bs64` # Assigned Experiment Name
  - **Goal**: Evaluate training performance (throughput) and model quality (loss, generation) with increased model capacity and context length on consumer hardware.
  - **Configuration**:
    - Model: ~1M parameters (`d_model=128, n_layers=6, n_head=8, d_hid=512`)
    - Context Length: 256 (`block_size=256`)
    - Tokenization: Character-level
    - Training: 5 epochs, `batch_size=64`, `gradient_accumulation_steps=1`
  - **Status**: 🟡 Starting Fresh (Resume path invalid/starting new run)
  - **Latest Checkpoint**: outputs/2025-03-29/19-28-11/checkpoints/checkpoint_step_8570.pt # Old path
  - **Context**: Reverted block_size from 1024 to 256 due to performance issues. This run uses batch_size 64 to evaluate performance against previous baselines after code refactoring.
  - **Metrics**: Track Tokens/sec, loss curves (step & epoch), qualitative generation results.

### 🔴 High Priority

*These tasks represent major foundational refactoring.*

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
    - ⏳ **Fix and Utilize TensorBoard Logging** (`TensorBoardLogger`)
      - **Status**: 🟡 Active (Debugging test failures - Paused pending basic workflow test)
      - 🔍 Previously identified as missing tests during review 2025-03-29
    - ⏳ Implement Tests for `SampleGenerationCallback`
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
    - ⏳ Investigate Model "Memory Architecture" (`docs/model.md` investigation)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Update Data Pipeline Documentation (`docs/data_pipeline.md`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Update Model Documentation (Parameter Count Difference) (`docs/model.md` update)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Verify `transformers` Dependency (Scheduler Usage)
      - 🔍 Identified during code review 2025-03-28

## Upcoming Tasks

- ### 🟡 Review Legacy Code
  - **Goal**: Review `src/` and `scripts/` directories to identify and remove or archive deprecated/legacy code (e.g., potentially unused utilities) after confirming functionality of refactored components (like `src/cli/run.py`).
  - **Status**: ⏳ To Do

- ### 🟠 Refactor Data Pipeline for Tokenizer Flexibility
  - **Goal**: Modify the data pipeline to support different tokenizers (character, subword, etc.) configured via Hydra, following the strategy outlined in `src/data/base.py`.
  - **Steps**: (See original task for detailed steps)
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

- [✅] **Complete Model Architecture Refactoring** (Est. Effort: High) ✅ 2025-03-30 (Assumed completed based on sub-tasks)
- ### 🟡 Implement Basic Unit Tests ✅ 2025-03-28
- ### 🟡 Implement Folder Structure ✅ 2025-03-28
- ### 🔴 Update Architecture Documentation ✅ 2025-03-28
- ### 🟡 Fix Config Defaults ✅ 2025-03-28
- ### 🔴 Integrate Proven Training Logic ✅ 2025-03-28
- ### 🟡 Complete Config Refactor (`configs` -> `conf`) ✅ 2025-03-28

### Data Processing Enhancements

-   **Task:** Implement Standard Tokenization in `src/data/processors.py`.
    -   **Status:** ✅ Done (2024-03-31)
    -   **Description:** Modify `prepare_text_data` (or add new logic) to load and use standard tokenizers (e.g., Hugging Face `transformers`/`tokenizers` like 'gpt2') based on configuration, saving token IDs instead of raw text. Save necessary tokenizer info (vocab size, special tokens).
    -   **Priority:** High
-   **Task:** Integrate Data Splitting in `src/data/processors.py`.
    -   **Status:** ✅ Done (2024-03-31)
    -   **Description:** Modify the data preparation logic (e.g., in `prepare_text_data`) to call the existing `split_data` function after loading/tokenizing, saving separate processed files for train, validation, and potentially test splits. Ensure configuration allows specifying split ratios.
    -   **Priority:** High
-   **Task:** Add Unit Tests for Tokenization and Splitting.
    -   **Status:** ✅ Done (2024-03-31)
    -   **Description:** Update `tests/unit/test_data_processors.py` to add tests covering the new standard tokenization logic (handling different tokenizers) and the integrated data splitting functionality.
    -   **Priority:** High