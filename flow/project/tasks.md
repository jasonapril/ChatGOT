# Tasks

*Part of the Flow System. See also: [Vision and Plan](vision_and_plan.md), [Guidelines](../system/guidelines.md).*

This file serves as the working memory for all active tasks in the project. It's designed to provide at-a-glance visibility into what's currently being worked on, what's coming up next, and what has recently been completed.

## Guidelines and Conventions [TODO: Rename this to legend or similar?]

- **Priority Colors**:
    -   ⚪ Placeholder
    -   🔴 Critical - Used for urgent items that block progress
    -   🟠 High - Important items that need immediate attention
    -   🟡 Medium - Standard work items
    -   🟢 Low - Nice-to-have items or long-term improvements

- **Status Indicators**:
    -   ⏳ In progress - Actively being worked on
    -   ✅ Completed - Work is finished and verified
    -   🔄 Under review - Awaiting feedback or assessment
    -   ⏸️ Paused - Temporarily suspended
    -   🚩 Blocked - Unable to proceed due to dependencies

## Task Template 2025-04-06 (Copy and paste to create new tasks)

### ⚪ Task Short Description

*   **Status:** TODO
*   **Source:** USER (YYYY-MM-DD) / AI (YYYY-MM-DD)
*   **Description:**
    *   Detailed description of the task.
*   **Context/Links:**
    *   [Link to relevant files/discussions](path)
*   **Acceptance Criteria:**
    *   How do we know this task is done?
*   **Plan/Sub-tasks:**
    *   [ ] Step 1
    *   [ ] Step 2
*   **Notes/Log:**
    *   *AI (YYYY-MM-DD HH:MM):* Initial entry.

-------------------------------------------------------------------------------

## Task List

### 🟡 Document Pydantic Role in Config Validation

*   **Status:** TODO
*   **Source:** AI (2025-04-06) - Based on user request
*   **Description:**
    *   Update `conf/README.md` to explain the role of Pydantic schemas (`src/craft/config/schemas.py`) in validating the configuration *after* it has been composed by Hydra/OmegaConf.
    *   Clarify that Hydra handles merging/composition/interpolation, while Pydantic enforces structure and types.
*   **Context/Links:**
    *   Conversation leading to this task.
    *   `conf/README.md`
    *   `src/craft/config/schemas.py`
*   **Acceptance Criteria:**
    *   `conf/README.md` clearly describes the two-stage config process (Hydra composition + Pydantic validation).
*   **Plan/Sub-tasks:**
    *   [ ] Draft explanation of Pydantic's role.
    *   [ ] Integrate explanation into `conf/README.md`.
    *   [ ] Review with USER.
*   **Notes/Log:**
    *   *AI (2025-04-06 15:18):* Task created as requested.

### 🟠 Refactor and Simplify Configuration Management

*   **Status:** Open
*   **Source:** AI (2025-04-06) - Based on user request
*   **Updated:** 2025-04-06
*   **Goal:** Improve the robustness, maintainability, and debuggability of the configuration system (Hydra + Pydantic). Reduce time spent debugging configuration-related issues.
*   **Description:** The current configuration system, while functional, led to significant debugging effort. This task involves a review and potential refactoring to simplify its usage and improve error handling. Key areas to investigate:
    1.  **Interpolation:** Review complex OmegaConf interpolations (e.g., `${experiment...}` within `@package _group_` contexts) and consider simpler alternatives like direct value setting or improved default inheritance to enhance robustness.
    2.  **Hydra Structured Configs:** Explore deeper integration with Hydra's Structured Configs (using dataclasses) potentially replacing or streamlining the current manual `OmegaConf.to_container -> Pydantic(**dict)` validation step for earlier, more integrated validation and potentially clearer errors.
    3.  **Debugging/Logging:** Enhance logging within `src/craft/main.py`'s config loading sequence and potentially Pydantic validation steps to provide more specific feedback on where errors occur.
    4.  **Component Interfaces:** Ensure components interacting based on config (e.g., data preparation scripts saving pickles and `Dataset` classes loading them) have robust and compatible interfaces, or that loaders provide clearer errors for format mismatches.
    5.  **Documentation:** Improve documentation (`conf/README.md`, related to T006) outlining the flow, conventions, and potential pitfalls of the configuration system.
*   **Links/References:** Conversation history leading to this task (specifically around E2E test debugging).
*   **Notes:** Consider tackling this after resolving the current E2E test blockers, although improvements here might prevent future similar issues.

### 🔴 **Restructure Source Code Organization** ⏳
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

### 🟠 Fix Hydra Configuration Loading
- **Status**: 🟡 To Do
- **Goal**: Ensure configuration overrides in files (e.g., `conf/training/minimal_test.yaml`) correctly override defaults without needing command-line arguments.
- **Context**: Currently relying on `data.block_size=256` override on command line for minimal test due to issues with YAML override precedence.
- **Update (2025-04-06)**: Encountered significant issues resolving defaults specified within experiment config files (`conf/experiment/*.yaml`). Seems necessary to use standard Hydra syntax (`- /group: name`) in experiment defaults, or omit them entirely if covered by main `config.yaml` defaults.

### 🟠 Improve Configuration Handling
  - **Sub-tasks**:
    - ⏳ Improve Configuration Validation (Pydantic/JSONSchema) (`src/config/config_manager.py`)
      - *Prerequisite for: Refactor Dataset Loading*
      - 🔍 Identified during code review 2025-03-28

### 🟠 Refactor Core Training Logic
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

### 🟠 Refactor Specific Components
- **Sub-tasks**:
  - ⏳ Remove Unused PositionalEncoding (`src/models/transformer.py`)
    - 🔍 Identified during code review 2025-03-28
  - ⏳ Clarify/Consolidate Model Implementations & Generation (`gpt_decoder.py`, `transformer.py`, `generation.py`)
    - *Contributes to: Complete Model Architecture Refactoring Goal*
    - 🔍 Identified during code review 2025-03-28

### 🟠 Improve Testing & CI
- **Status**: 🟡 To Do
- **Sub-tasks**:
  - ⏳ **Fix and Utilize TensorBoard Logging** ✅ 2025-04-03
    - **Goal**: Ensure TensorBoard logs are generated correctly, easily accessible, and provide meaningful insights. Fix path issues and integrate with training loop.
    - **Context**: Was generating logs in nested `lightning_logs` directories. Refactored `TensorBoardLogger`. Verified basic logging and resume functionality.
    - **Status**: ✅ Done
  - [x] Implement Tests for `SampleGenerationCallback` ✅ <# Done 2025-04-06 (Added init, prompt source, error handling tests) #>
  - ⏳ Refactor Test Structure and Runner (`tests/run_all_tests.py`, directory org)
  - ⏳ Fix Test Import Handling (Remove `try-except-mock`)
  - ⏳ **Improve Generation Test Coverage**: 🟡 Active
    - **Goal**: Enhance `test_generation.py` to cover more generation scenarios, including different sampling parameters (temperature, top-k) and prompt variations.
    - **Context**: Current test is basic. Need to ensure generation quality and robustness under various conditions.
    - **Updates**:
      - Fixed several persistent test failures in `test_generation.py` and `test_standalone_generation.py` (vocab issues, test expectations). (Apr 1)
      - Added edge case tests for `batch_generate` (empty prompt, max_length=0). (Apr 1)
      - [x] Add tests for `src/craft/utils/generation.py`. ✅ <# Done 2025-04-06 (Verified 100% coverage) #>
    - 🔍 Identified during code review 2025-03-28
  - ⏳ Refactor Data Test Configuration (`DataTests`)
  - ⏳ Fix Integration Tests (`tests/integration_tests.py`)
  - ⏳ Set up CI/CD pipeline (GitHub Actions)
  - ⏳ **Evaluate Smoke Tests and Expand Integration Tests**: 🟡 To Do
    - **Goal**: Assess the need for a dedicated training smoke test and/or a unified smoke test. Review and expand the existing integration and feature tests for broader coverage.
    - **Context**: Current smoke tests focus on generation. A quick training check and more comprehensive integration tests would improve robustness.
    - **Update (2025-04-06)**: Implemented basic E2E integration tests (`tests/training/integration/test_e2e_training.py`) using dedicated test experiment configs (`test_got_char.yaml`, `test_got_subword.yaml`) that run short training loops and check for outputs (logs, checkpoints).
  - 🔍 Planned

### 🟡 Review Craft Directory READMEs
- **Status**: 🟡 To Do
- **Goal**: Ensure key directories within the Craft project (`src/`, `tests/`, `docs/`, `conf/`, etc.) have shallow README.md files serving as descriptive indexes, following Flow principles.

### 🟠 Review `conf` Directory
- **Status**: Upcoming
- Standardize. Ensure good defaults. Maintain a separation between configs used for tests, experiments, etc. What should the major categories be?
- We may need a better way to manage these configs.

### 🟠 Add Ability to Train on MNIST
- **Status**: Upcoming
- This marks a major transition to multimodality. This will also ensure a higher degree of modularity.

### 🔴 Decide What to do with Completed Tasks
- **Status**: Upcoming
- Probably keep them in a log.

### 🔴 Define Goals
- **Status**: Upcoming
- We should define the precise goals that we wish to attain with this project, and work backwards from there to ensure that the project's implementations are ideal for fulfilling those goals.
- Develop AI models that work on low-spec devices (performance critical)
- Experiment with cutting edge AI architectures

### 🟠 Review Dependencies
- **Status**: Upcoming
- Which libraries/frameworks/packages/etc. does Craft depend on? Are they all justified and ideal? What are the pros and cons compared to alternatives? When does it make sense to use custom libraries/frameworks/packages/etc.?
- **Specific Checks**:
  - Verify `transformers` Dependency (specifically for Scheduler Usage) - is it necessary? (Identified during code review 2025-03-28)

### 🟡 Improve CLI Generation Tokenization
- **Status**: Upcoming
- Refactor `src/cli/run.py::generate_text`

### 🟠 Train Small Model on GoT (Subword)
- **Status**: 🟢 Active <# Resuming attempt for 95M param model variant on 2025-04-06. Running with activation checkpointing. #>
- **Goal**: Perform a training run using a small model variant (configured for low-spec devices) on the Game of Thrones Season 1 dataset, utilizing a subword tokenizer.
- **Context**: This serves as an initial test case for the refactored data pipeline (tokenizer flexibility) and aligns with the goal of optimizing for low-spec devices. Requires GoT dataset prepared with a subword tokenizer (e.g., BPE or SentencePiece).
- **Depends on**: Refactor Data Pipeline for Tokenizer Flexibility

### CI/CD & Deployment
- **Status**: Future
  - [ ] Set up basic CI (GitHub Actions?) to run tests on push.
  - [ ] Explore deployment options (e.g., containerization, cloud services).

### Configuration Logic
- **Subtasks**:
  - [ ] Improve Configuration Handling (`conf/`, Hydra instantiation, Pydantic validation).
  - [x] Integrate basic Pydantic validation in `scripts/train.py` (2025-04-03)
    - Files: `scripts/train.py`, `src/craft/config/schemas.py`, `src/craft/data/base.py`, `src/craft/models/factory.py`, `src/craft/training/optimizers.py`, `src/craft/training/schedulers.py`

### Refactoring & Cleanup (Ongoing)
- **Subtasks**:
   - [ ] Address TODOs and FIXMEs in the code.
   - [ ] Improve test coverage (currently 81%).
   - [ ] Standardize docstrings (NumPy style?).
   - [ ] Add type hints where missing.
   - [ ] Refactor `tests/training/test_amp.py` tests for `SafeGradScaler` after recent updates. <# Added 2024-04-03 #>
   - [ ] **Refactor Large Callback Test Files:** 🟡 Active <# Added 2025-04-04 #>
     - **Goal**: Break down large test files in `tests/training/callbacks/` (e.g., `test_early_stopping.py`, `test_sample_generation.py`) into smaller, more focused test classes/files for improved readability and maintainability.
     - **Context**: Identified after refactoring callback source code and fixing associated tests.
     - **Update (2025-04-06):** Reviewed `test_early_stopping.py`; structure seems reasonable, no major refactoring needed currently. `test_sample_generation.py` reviewed and improved.
   - [ ] **Evaluate and Refactor Large Files:** 🟡 Active <# Started 2025-04-04 #>
     - **Goal**: Assess files like `src/craft/training/callbacks.py` and `src/craft/data/base.py` for potential refactoring opportunities (e.g., splitting into smaller modules) to improve readability and maintainability.

### Feature Enhancements (Ideas)
- **Subtasks**:
   - [ ] Add support for different model architectures (RNN, etc.).
   - [ ] Add support for loading and generating with third-party models (e.g., from Hugging Face Hub) in `scripts/generate.py`.
   - [ ] Add support for third-party models in general

### **[P2 - Medium]** Task: Improve test coverage for configuration loading and component instantiation #testing #config #ci
*   Status: `🟢 Active`
*   Goal: Prevent runtime errors related to Hydra config, Pydantic validation, and component setup (factories, callbacks).
*   Details:
    *   Add integration tests for `main.py` using `hydra.experimental.compose`.
    *   Add tests for factory functions with realistic `DictConfig`.
    *   Add tests verifying callback instantiation loop.
    *   Add tests for checkpoint save/resume.

### **[P1 - High]** Task: Resume training chatgot_95m_subword (Subword Tokenizer)
*   Status: `🚧 Blocked` (Blocked by configuration/runtime errors)

-------------------------------------------------------------------------------

## Completed Tasks (Move to Log)

### 🟠 Refactor Data Pipeline for Tokenizer Flexibility
- **Status**: ✅ Done <# 2025-04-05: Verified structure and configuration supports flexibility. Tokenizer artifacts moved to outputs/ #>
- **Goal**: Modify the data pipeline to support different tokenizers (character, subword, etc.) configured via Hydra, following the strategy outlined in `src/data/base.py`.
- **Steps**: (See original task for detailed steps)

### 🟡 Review Legacy Code
- **Status**: ✅ 2025-04-01
- **Goal**: Review `src/` and `scripts/` directories to identify and remove or archive deprecated/legacy code (e.g., potentially unused utilities) after confirming functionality of refactored components (like `src/cli/run.py`).

### Test Output Cleanup
- **Status**: ✅ 2025-04-03
- **Goal**: Fix issue where tests create persistent top-level directories.
- **Context**: Previously labeled Task 3.3.
- **Files**: `tests/data/tokenizers/test_subword.py`, `tests/data/tokenizers/test_sentencepiece.py`, `tests/training/test_checkpointing.py`

### 🔴 Debug E2E Training Failures via Staged Testing

*   **Status:** ✅ Completed (2025-04-06)
*   **Source:** AI/USER (2025-04-06) - Response to persistent E2E test failures.
*   **Description:** The E2E tests (`test_e2e_training.py`) were consistently failing with various errors. A structured, stage-based testing approach was used to isolate the root causes.
*   **Context/Links:**
    *   Conversation history leading to this task.
    *   `tests/training/integration/test_e2e_training.py`
    *   `tests/training/integration/test_main_stages.py` (Stages 1 & 2 implemented)
    *   `src/craft/main.py`
*   **Acceptance Criteria:**
    *   [x] Stages 1 & 2 pass for both `test_got_char` and `test_got_subword` configurations.
    *   [x] Root cause of the original E2E failures is identified and fixed.
    *   [x] E2E tests in `test_e2e_training.py` pass reliably.
*   **Plan/Sub-tasks:** (Stages 1 & 2 implemented in `tests/training/integration/test_main_stages.py`)
    *   [x] **Stage 1: Config Loading & Validation:** Test Hydra composition + Pydantic validation.
    *   [x] **Stage 2: Dataloader Preparation:** Test tokenizer/dataset loading and DataLoader creation.
    *   [x] **Stage 3: Model & Optimizer/Scheduler Instantiation:** Test component creation (Covered by E2E test).
    *   [x] **Stage 4: Callback Instantiation:** Test callback creation (Covered by E2E test).
    *   [x] **Stage 5: Trainer Initialization:** Test `Trainer` and `CheckpointManager` initialization (Covered by E2E test).
    *   [x] **Stage 6: Minimal Training Step & Checkpointing Logic:** Test single step + `save_checkpoint` (Covered by E2E test).
    *   [x] Fix underlying issues identified during staged testing (tokenizer loading, main.py call, CheckpointManager marker, TensorBoard callback signature).
    *   [x] Rerun and verify `test_e2e_training.py` passes.
*   **Notes/Log:**
    *   *AI (2025-04-06 16:39):* Task created. Starting implementation of Stage 1.
    *   *AI (2025-04-06 ~17:00):* Implemented Stage 1 & 2 tests. Debugged multiple issues related to tokenizer loading (paths, classmethod, config saving), main script config passing, checkpoint marker suffix, and TensorBoard callback signature.
    *   *AI (2025-04-06 17:05):* E2E tests now pass for both configurations, implicitly verifying stages 3-6. Removed redundant Stage 2 test, kept Stage 1 as a quick config check. Task completed.

### Complete Model Architecture Refactoring
- **Status**: ✅ 2025-03-30 (Assumed completed based on sub-tasks)

### 🟡 Implement Basic Unit Tests
- **Status**: ✅ 2025-03-28

### 🟡 Add Integration Test for Data Preparation

*   **Status:** ✅ 2025-04-06
*   **Source:** AI (2025-04-06) - Based on test coverage analysis
*   **Description:** Create an integration test (`tests/data/integration/test_prepare_data.py`?) that runs `scripts/prepare_data.py` as a subprocess for both character and subword tokenization types. The test should verify successful execution (exit code 0) and the creation of expected output files (e.g., `.pkl` files, tokenizer artifacts) in a temporary directory.
*   **Context/Links:**
    *   Conversation discussing test coverage.
    *   `scripts/prepare_data.py`
    *   `tests/data/integration/test_prepare_data.py`
*   **Acceptance Criteria:**
    *   [x] Test runs successfully via pytest.
    *   [x] Test covers both 'char' and 'subword' preparation using minimal configs/data.
    *   [x] Test asserts successful script completion and existence of expected output files.
*   **Plan/Sub-tasks:**
    *   [x] Create test file and basic structure.
    *   [x] Define fixtures for script path, sample input data, temp output dir.
    *   [x] Parameterize test for 'char' and 'subword' types.
    *   [x] Implement subprocess call.
    *   [x] Add assertions for exit code and file existence.
*   **Notes/Log:**
    *   *AI (2025-04-06 17:21):* Task created.
    *   *AI (2025-04-06 17:29):* Implemented test, fixed argument parsing issues, verified test passes.

### 🟡 Add Integration Test for CLI Generation

*   **Status:** ✅ 2025-04-06
*   **Source:** AI (2025-04-06) - Based on test coverage analysis
*   **Description:** Create an integration test (`tests/cli/integration/test_cli_generate.py`?) that runs the `generate` command via `src/cli/run.py` as a subprocess. The test needs a small, known checkpoint to load from. It should verify successful execution (exit code 0) and that some text is printed to stdout.
*   **Context/Links:**
    *   Conversation discussing test coverage.
    *   `src/cli/run.py`
*   **Acceptance Criteria:**
    *   [x] Test runs successfully via pytest.
    *   [x] Test uses a minimal checkpoint.
    *   [x] Test asserts successful script completion and non-empty text output.
*   **Plan/Sub-tasks:**
    *   [x] Determine strategy for providing a minimal checkpoint (generate dynamically? check one in?). -> Generated one via 1-step training, checked in relevant files.
    *   [x] Create test file and basic structure.
    *   [x] Define fixtures/setup for script path, checkpoint path, prompt.
    *   [x] Implement subprocess call for the `generate text` command (using `-m craft.cli.run`).
    *   [x] Add assertions for exit code and stdout content.
*   **Notes/Log:**
    *   *AI (2025-04-06 17:21):* Task created.
    *   *AI (2025-04-06 ~21:38):* Created minimal checkpoint artifact. Debugged numerous issues in `generate_commands.py` related to CLI argument parsing (`--checkpoint`), config loading (prioritizing `config.yaml`, fixing nested structure), tokenizer loading (`create_tokenizer`, classmethod assignment), `vocab_size` injection, and `CharTokenizer.encode` signature. Test `tests/cli/integration/test_cli_generate.py` now passes.

## New Tasks Always Seem to be Added at the End, So Review and Sort These
### 🟡 Implement and Test CLI Evaluate Command

*   **Status:** TODO
*   **Source:** AI (2025-04-06)
*   **Description:** Implement the `evaluate model` command logic within `src/craft/cli/evaluate_commands.py`. This should involve loading a model checkpoint, loading evaluation data, running the evaluation loop (likely using `craft.training.evaluation.Evaluator`), and reporting metrics. Create an associated integration test (`tests/cli/integration/test_cli_evaluate.py`) that runs the command as a subprocess and verifies its successful execution and output.
*   **Context/Links:**
    *   `src/craft/cli/evaluate_commands.py` (Currently stubbed)
    *   `src/craft/training/evaluation.py`
*   **Acceptance Criteria:**
    *   `evaluate model` command is implemented.
    *   Integration test passes, verifying successful execution and metric reporting (or log output indicating metrics).
*   **Plan/Sub-tasks:**
    *   [ ] Define necessary arguments for `evaluate model` (checkpoint path, data config/path, metrics, etc.).
    *   [ ] Implement loading logic for model, data, and potentially config.
    *   [ ] Integrate with `Evaluator` class.
    *   [ ] Implement metric reporting to console/log.
    *   [ ] Create integration test file (`test_cli_evaluate.py`).
    *   [ ] Add fixtures for checkpoint, data, etc.
    *   [ ] Implement subprocess call and assertions.
*   **Notes/Log:**
    *   *AI (2025-04-06 21:46):* Task created. Evaluation command is currently unimplemented.