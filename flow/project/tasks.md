# Tasks

*Part of the Flow System. See also: [Vision and Plan](vision_and_plan.md), [Guidelines](../system/guidelines.md), [Improvements](../planning/improvements.md).*

This file serves as the working memory for all active tasks in the project. It's designed to provide at-a-glance visibility into what's currently being worked on, what's coming up next, and what has recently been completed.

## Active Tasks

### 🔴 High Priority

*These tasks represent major foundational refactoring.*

- **Restructure Source Code Organization**: 🟡 ⏳
  - **Goal**: Implement a cleaner, more maintainable source code structure by relocating utilities and removing unused/non-standard modules.
  - **Previous Structure**:
    ```
    src/craft/
    ├── cli/
    ├── config/
    ├── data/
    ├── models/
    ├── performance/  <- To be removed
    ├── training/
    └── utils/        <- To be cleaned up
    ```
  - **Target Structure**:
    ```
    src/craft/
    ├── cli/
    ├── config/
    ├── data/
    ├── models/
    ├── training/     <- Will contain checkpoint_utils, metrics, memory_utils
    └── utils/        <- Will contain logging, io, generation, common
    ```
  - **Implementation Plan (Revised 2025-04-03)**:
    1. **Preparation Phase**:
       - [x] Review all existing tests and their coverage ✅ <# Done 2025-04-03 #>
       - [x] Document current import dependencies ✅ <# Done 2025-04-03 (see flow/project/dependencies.dot) #>
       - [ ] Create test suite structure for target layout (Minimal changes needed now)
       - [ ] Set up CI to run tests on each change (Skipping for now)

    2. **Code Migration & Cleanup Phase**: ✅ <# Done 2025-04-03 #>
       - [x] **Move `utils/checkpoint.py`** -> `training/checkpoint_utils.py`. Verify imports. ✅ <# Done 2025-04-03 #>
       - [x] **Move `utils/metrics.py`** -> `training/metrics.py`. Verify imports. ✅ <# Done 2025-04-03 #>
       - [x] **Move `utils/memory.py`** -> `training/memory_utils.py`. Update import in `training/optimizations.py`. ✅ <# Done 2025-04-03 #>
       - [x] **Delete `utils/performance.py`** (Unused). ✅ <# Done 2025-04-03 #>
       - [x] **Delete `performance/` directory** (Unused/Experimental). ✅ <# Done 2025-04-03 #>
       - [x] **Review `utils/common.py`:** Relocate/remove functions, improve coverage. ✅ <# Done 2025-04-03 #>
       - [x] Run full test suite after each significant step. ✅ <# Done 2025-04-03 #>

    3. **Final Steps**:
       - [x] Refactor `Model.save/load`, fix `test_save_load`, merge `checkpoint_utils.py` into `checkpointing.py`. ✅ <# Done 2025-04-03 #>
       - [ ] Update documentation (if affected beyond tasks.md).
       - [ ] Clean up any remaining old files/artifacts.
       - [x] Verify all functionality works (e.g., run a sample training/generation). ✅ <# Done 2025-04-03
        #>

  - **Testing Strategy**:
    - Run `pytest` after each file move/deletion to catch immediate issues.
    - Improve coverage for modified/relocated modules as needed (esp. `common.py`).

  - **Priority**: High (Foundational Cleanup)
  - **Status**: In Progress
  - 🔍 Original task identified during code review 2025-03-31
  - 📝 Plan revised based on analysis 2025-04-03

### 🟠 Medium Priority

*These tasks improve existing components or address known gaps.*

- #### Fix Hydra Configuration Loading
  - **Goal**: Ensure configuration overrides in files (e.g., `conf/training/minimal_test.yaml`) correctly override defaults without needing command-line arguments.
  - **Context**: Currently relying on `data.block_size=256` override on command line for minimal test due to issues with YAML override precedence.
  - **Status**: 🟡 To Do

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
    - ⏳ Implement and Verify Checkpointing ✅ 2025-04-03
      - **Goal**: Implement model checkpoint saving (model state, optimizer state, epoch/step) periodically (e.g., based on steps or validation performance) and allow resuming training from a specified checkpoint.
      - **Context**: Essential for long training runs. Builds on the working basic training loop. Checkpoint location handled manually via `outputs/runs/<timestamp>/checkpoints/`.
      - **Status**: ✅ Done

- #### Refactor Specific Components
  - **Sub-tasks**:
    - ⏳ Remove Unused PositionalEncoding (`src/models/transformer.py`)
      - 🔍 Identified during code review 2025-03-28
    - ⏳ Clarify/Consolidate Model Implementations & Generation (`gpt_decoder.py`, `transformer.py`, `generation.py`)
      - *Contributes to: Complete Model Architecture Refactoring Goal*
      - 🔍 Identified during code review 2025-03-28

- #### Improve Testing & CI
      - **Sub-tasks**:
        - ⏳ **Fix and Utilize TensorBoard Logging** ✅ 2025-04-03
          - **Goal**: Ensure TensorBoard logs are generated correctly, easily accessible, and provide meaningful insights. Fix path issues and integrate with training loop.
          - **Context**: Was generating logs in nested `lightning_logs` directories. Refactored `TensorBoardLogger`. Verified basic logging and resume functionality.
          - **Status**: ✅ Done
        - ⏳ Implement Tests for `SampleGenerationCallback`
        - ⏳ Refactor Test Structure and Runner (`tests/run_all_tests.py`, directory org)
        - ⏳ Fix Test Import Handling (Remove `try-except-mock`)
        - ⏳ **Improve Generation Test Coverage**: 🟡 Active
          - **Goal**: Enhance `test_generation.py` to cover more generation scenarios, including different sampling parameters (temperature, top-k) and prompt variations.
          - **Context**: Current test is basic. Need to ensure generation quality and robustness under various conditions.
          - **Updates**:
            - Fixed several persistent test failures in `test_generation.py` and `test_standalone_generation.py` (vocab issues, test expectations). (Apr 1)
            - Added edge case tests for `batch_generate` (empty prompt, max_length=0). (Apr 1)
            - **Next**: Add tests for `src/craft/utils/generation.py` (currently 0% coverage).
          - 🔍 Identified during code review 2025-03-28
        - ⏳ Refactor Data Test Configuration (`DataTests`)
        - ⏳ Fix Integration Tests (`tests/integration_tests.py`)
        - ⏳ Set up CI/CD pipeline (GitHub Actions)
        - ⏳ **Evaluate Smoke Tests and Expand Integration Tests**: 🟡 To Do
          - **Goal**: Assess the need for a dedicated training smoke test and/or a unified smoke test. Review and expand the existing integration and feature tests for broader coverage.
          - **Context**: Current smoke tests focus on generation. A quick training check and more comprehensive integration tests would improve robustness.
          - **Status**: 🟡 To Do
        - 🔍 Planned

- ### 🟡 Review Craft Directory READMEs
  - **Goal**: Ensure key directories within the Craft project (`src/`, `tests/`, `docs/`, `conf/`, etc.) have shallow README.md files serving as descriptive indexes, following Flow principles.
  - **Status**: 🟡 To Do
  - **Priority**: Low

## Upcoming Tasks

- ### 🟠 Review `conf` Directory
  - Standardize. Ensure good defaults. Maintain a separation between configs used for tests, experiments, etc. What should the major categories be?
  - We may need a better way to manage these configs.

- ### 🟠 Add Ability to Train on MNIST
  - This marks a major transition to multimodality. This will also ensure a higher degree of modularity.

- ### 🔴 Decide What to do with Completed Tasks
  - Probably keep them in a log.

- ### 🟠 Refactor Data Pipeline for Tokenizer Flexibility
  - **Goal**: Modify the data pipeline to support different tokenizers (character, subword, etc.) configured via Hydra, following the strategy outlined in `src/data/base.py`.
  - **Steps**: (See original task for detailed steps)
  - **Status**: ✅ Done <# 2025-04-05: Verified structure and configuration supports flexibility. Tokenizer artifacts moved to outputs/ #>

- ### 🔴 Define Goals
  - We should define the precise goals that we wish to attain with this project, and work backwards from there to ensure that the project's implementations are ideal for fulfilling those goals.
  - Develop AI models that work on low-spec devices (performance critical)
  - Experiment with cutting edge AI architectures

- ### 🟠 Review Dependencies
  - Which libraries/frameworks/packages/etc. does Craft depend on? Are they all justified and ideal? What are the pros and cons compared to alternatives? When does it make sense to use custom libraries/frameworks/packages/etc.?
  - **Specific Checks**:
    - Verify `transformers` Dependency (specifically for Scheduler Usage) - is it necessary? (Identified during code review 2025-03-28)

- ### 🟡 Improve CLI Generation Tokenization
  - Refactor `src/cli/run.py::generate_text`

- ### 🟠 Train Small Model on GoT S1 (Subword)
  - **Goal**: Perform a training run using a small model variant (configured for low-spec devices) on the Game of Thrones Season 1 dataset, utilizing a subword tokenizer.
  - **Context**: This serves as an initial test case for the refactored data pipeline (tokenizer flexibility) and aligns with the goal of optimizing for low-spec devices. Requires GoT S1 dataset prepared with a subword tokenizer (e.g., BPE or SentencePiece).
  - **Depends on**: Refactor Data Pipeline for Tokenizer Flexibility
  - **Status**: 🟡 Active <# Resuming attempt for 95M param model variant on 2025-04-05. Previously started 1M param model. #>
  - **Priority**: Medium (once dependency met)

### CI/CD & Deployment (Future)
   - [ ] Set up basic CI (GitHub Actions?) to run tests on push.
   - [ ] Explore deployment options (e.g., containerization, cloud services).

### Configuration Logic
   - [ ] Improve Configuration Handling (`conf/`, Hydra instantiation, Pydantic validation).
   - [x] Integrate basic Pydantic validation in `scripts/train.py` (2025-04-03)
     - Files: `scripts/train.py`, `src/craft/config/schemas.py`, `src/craft/data/base.py`, `src/craft/models/factory.py`, `src/craft/training/optimizers.py`, `src/craft/training/schedulers.py`

### Refactoring & Cleanup (Ongoing)
   - [ ] Address TODOs and FIXMEs in the code.
   - [ ] Improve test coverage (currently 81%).
   - [ ] Standardize docstrings (NumPy style?).
   - [ ] Add type hints where missing.
   - [ ] Refactor `tests/training/test_amp.py` tests for `SafeGradScaler` after recent updates. <# Added 2024-04-03 #>
   - [ ] **Refactor Large Callback Test Files:** 🟡 To Do <# Added 2025-04-04 #>
     - **Goal**: Break down large test files in `tests/training/callbacks/` (e.g., `test_early_stopping.py`, `test_sample_generation.py`) into smaller, more focused test classes/files for improved readability and maintainability.
     - **Context**: Identified after refactoring callback source code and fixing associated tests.
   - [ ] **Evaluate and Refactor Large Files:** 🟡 Active <# Started 2025-04-04 #>
     - **Goal**: Assess files like `src/craft/training/callbacks.py` and `src/craft/data/base.py` for potential refactoring opportunities (e.g., splitting into smaller modules) to improve readability and maintainability.

### Feature Enhancements (Ideas)
   - [ ] Add support for different model architectures (RNN, etc.).
   - [ ] Add support for loading and generating with third-party models (e.g., from Hugging Face Hub) in `scripts/generate.py`.
   - [ ] Add support for third-party models in general

## Completed Tasks (Recent)

- ### 🟡 Review Legacy Code ✅ 2025-04-01
  - **Goal**: Review `src/` and `scripts/` directories to identify and remove or archive deprecated/legacy code (e.g., potentially unused utilities) after confirming functionality of refactored components (like `src/cli/run.py`).

# Adding previously known completed tasks that seem missing
- [✅] **Complete Model Architecture Refactoring** (Est. Effort: High) ✅ 2025-03-30 (Assumed completed based on sub-tasks)
- ### 🟡 Implement Basic Unit Tests ✅ 2025-03-28

### Test Output Cleanup ✅ 2025-04-03
  - **Goal**: Fix issue where tests create persistent top-level directories.
  - **Context**: Previously labeled Task 3.3.
  - **Status**: DONE 2025-04-04
  - **Files**: `tests/data/tokenizers/test_subword.py`, `tests/data/tokenizers/test_sentencepiece.py`, `tests/training/test_checkpointing.py`