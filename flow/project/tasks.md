# Tasks

*Part of the Flow System. See also: [Vision and Plan](vision_and_plan.md), [Guidelines](../system/guidelines.md), [Improvements](../planning/improvements.md).*

This file serves as the working memory for all active tasks in the project. It's designed to provide at-a-glance visibility into what's currently being worked on, what's coming up next, and what has recently been completed.

## Active Tasks

### 🔴 High Priority

*These tasks represent major foundational refactoring.*

- **Restructure Source Code Organization**: 🟡 ⏳
  - **Goal**: Implement a flatter, more maintainable source code structure following industry standards
  - **Current Structure**:
    ```
    src/
    ├── cli/
    ├── data/
    ├── models/
    ├── training/
    ├── performance/
    ├── utils/
    └── __init__.py
    ```
  - **Proposed Structure**:
    ```
    src/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── base.py
    │   └── transformer.py
    ├── data/
    │   ├── __init__.py
    │   ├── processors.py
    │   └── tokenizers.py
    ├── training.py
    ├── utils.py
    └── cli.py
    ```
  - **Implementation Plan**:
    1. **Preparation Phase**:
       - [x] Review all existing tests and their coverage ✅ <# Done Apr 04 #>
       - [ ] Document current import dependencies 🟡 <# In Progress #>
       - [ ] Create test suite for new structure
       - [ ] Set up CI to run tests on each change
    
    2. **Code Migration Phase** (for each module):
       - [ ] Create new file structure
       - [ ] Move code piece by piece
       - [ ] Update imports
       - [ ] Run tests
       - [ ] Fix any issues
       - [ ] Only remove old code after tests pass
    
    3. **Module-Specific Tasks**:
       - **Models Module**:
         - [ ] Move `base.py` and `transformer.py`
         - [ ] Update model factory
         - [ ] Test model creation and initialization
       
       - **Data Module**:
         - [ ] Move processors and tokenizers
         - [ ] Update dataset loading
         - [ ] Test data pipeline end-to-end
       
       - **Training Module**:
         - [ ] Move core training logic
         - [ ] Update training callbacks
         - [ ] Test training loop
       
       - **Utils Module**:
         - [ ] Move common utilities
         - [ ] Update logging
         - [ ] Test utility functions
       
       - **CLI Module**:
         - [ ] Move command handlers
         - [ ] Update command registration
         - [ ] Test CLI commands
    
    4. **Final Steps**:
       - [ ] Run full test suite
       - [ ] Update documentation
       - [ ] Clean up old files
       - [ ] Verify all functionality works
    
  - **Testing Strategy**:
    - Unit tests for each module
    - Integration tests for module interactions
    - End-to-end tests for complete workflows
    - Performance tests for critical paths
    
  - **Priority**: High (Foundational)
  - **Status**: In Progress
  - 🔍 Identified during code review 2025-03-31
  - 📝 Updated with detailed implementation plan 2025-03-31

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
        - 🔍 Planned

- ### 🟡 Review Craft Directory READMEs
  - **Goal**: Ensure key directories within the Craft project (`src/`, `tests/`, `docs/`, `conf/`, etc.) have shallow README.md files serving as descriptive indexes, following Flow principles.
  - **Status**: 🟡 To Do
  - **Priority**: Low

## Upcoming Tasks

- ### 🟠 Review Flow Alignment & Identify System Improvements
  - **Goal**: Ensure development practices align with Flow principles and improve the Flow system's effectiveness based on recent work.
  - **Sub-tasks**:
    - [x] Review recent debugging work (test fixing) against Flow principles (`flow/flow.md`). ✅
    - [ ] Review recent debugging work against Flow protocol (`flow/flow.md`, `flow/gemini-2.5-pro-exp-03-25.md`), especially `tasks.md` update adherence. ⏳
    - [ ] Review `flow/meta/` contents for relevance, cruft, and potential refinements.
    - [ ] Review `system/guidelines_and_conventions.md` for any necessary updates discovered during the review.
    - [ ] Brainstorm and document specific improvements for the Flow system (e.g., `tasks.md` update triggers, definition of "significant step", logging integration, periodic checks).
    - [ ] Check `flow/domains/` for potential additions based on recent work (e.g., `pytest_debugging.md`?).
  - **Status**: 🟡 To Do
  - **Priority**: Medium

- ### 🟠 Refactor Data Pipeline for Tokenizer Flexibility
  - **Goal**: Modify the data pipeline to support different tokenizers (character, subword, etc.) configured via Hydra, following the strategy outlined in `src/data/base.py`.
  - **Steps**: (See original task for detailed steps)
  - **Status**: ⏳ To Do (Postponed from previous attempt)

- ### 🔴 Define Goals
  - We should define the precise goals that we wish to attain with this project, and work backwards from there to ensure that the project's implementations are ideal for fulfilling those goals.
  - Develop AI models that work on low-spec devices (performance critical)
  - Experiment with cutting edge AI architectures (research)

- ### 🟠 Review Dependencies
  - Which libraries/frameworks/packages/etc. does Craft depend on? Are they all justified and ideal? What are the pros and cons compared to alternatives? When does it make sense to use custom libraries/frameworks/packages/etc.?
  - **Specific Checks**:
    - Verify `transformers` Dependency (specifically for Scheduler Usage) - is it necessary? (Identified during code review 2025-03-28)

- ### 🟡 Improve CLI Generation Tokenization
  - Refactor `src/cli/run.py::generate_text`

## Completed Tasks (Recent)

- ### 🟡 Review Legacy Code ✅ 2025-04-01
  - **Goal**: Review `src/` and `scripts/` directories to identify and remove or archive deprecated/legacy code (e.g., potentially unused utilities) after confirming functionality of refactored components (like `src/cli/run.py`).

- [✅] **Complete Model Architecture Refactoring** (Est. Effort: High) ✅ 2025-03-30 (Assumed completed based on sub-tasks)
- ### 🟡 Implement Basic Unit Tests ✅ 2025-03-28