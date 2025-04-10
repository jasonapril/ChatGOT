# Craft AI Agent Context & Project Overview (README.md)

**Version:** <!-- AI Note: Agent updates this --> YYYY-MM-DD HH:MM UTC
**Last Updated By:** <!-- AI Note: Agent updates this --> <!-- TODO: Why do we need this? -->

---

**Purpose of this Document:**
This file serves as the **primary context and shared memory** for the Craft project, intended for both human developers and AI agents collaborating on its development <!-- TODO: It's primarily for the AI agent. -->. It outlines:
*   The project's vision, goals, and core principles. <!-- TODO: This is decided soley by the user, unless much of the rest. Maybe this helps us understand the role of the user compared to the agent? -->
*   The current architecture and technology choices (and the rationale behind them).
*   Key workflows for interacting with the framework.
*   Current development status, immediate tasks, and roadmap.
*   Development standards and guidelines.
*   Areas for future review and refinement.

**Nature of this Document:** This document is both:
*   **Descriptive:** It describes the current state, architecture, and capabilities of Craft (Sections 1-4, parts of 6).
*   **Prescriptive:** It guides current development efforts, outlining tasks and standards (Section 5, parts of 6).

**Future Evolution:** As the project grows, sections of this document may be split into separate files (e.g., `ARCHITECTURE.md`, `CONTRIBUTING.md`). However, this `README.md` will remain the central entry point, linking to more detailed documents as needed.

**Instructions for AI Agent:**
1.  **Ingest:** Read this document thoroughly at the start of each session.
2.  **Collaborate & Execute:** Your primary role is to actively assist in the development process by performing analysis, generating code, writing tests, refactoring, and updating documentation based on the objectives outlined here and specific instructions given in our chat. Assume responsibility for executing implementation tasks whenever feasible, seeking clarification or strategic direction when needed. <!-- TODO: Add planning? -->
3.  **Prioritize:** Focus on the **Current Task** (Section 5).
4.  **Update:** Keep **Section 5** (Status, Tasks, Progress) updated. Update the **Last Updated Date** and **Last Updated By** fields above when you modify Section 5. <!-- TODO: Even with this instruction listed twice, the AI has a tendency to quickly forget. -->
5.  **Reference:** Use this document and linked files (`conf/README.md`, etc.) to understand context. <!-- TODO: This may need to be more explicit. When exactly do we look at this other README files? -->
6.  **Align:** Ensure development tasks align with **Project Goals** and **Core Principles** (Section 1 & 2). Raise potential misalignments for discussion. <!-- TODO: This may need to be more explicit. Ensure how? -->
7.  **Ask:** If anything is unclear, requires more detail, or needs reconfirmation, please ask.

---

## 1. Project Overview: Craft

Craft is envisioned as a highly **modular and versatile** framework for the rapid **prototyping, training, evaluation, and deployment** of a diverse range of AI models.

**Core Idea:** To empower research and development by separating experimental configuration from core code execution, enabling flexible exploration of architectures and techniques with an emphasis on reproducibility, performance, and adaptability. <!-- TODO: There's a little more to the core idea, such as the ability to develop AI models with and for low-spec machines, and the ability to experiment with new and custom methods and architectures. Experimentation on low-spec machines is fundamental to the vision, even though Craft isn't meant to do this exclusively. -->

**Key Goals:**
*   **Modularity & Extensibility:** Design components (models, data handlers, training loops, callbacks) as interchangeable modules, primarily configured via external files.
*   **Configuration Driven:** Leverage a powerful configuration system to define experiments, promoting reproducibility and minimizing code changes for common variations.
*   **Versatility & Innovation:** Support a wide array of AI architectures (Transformers, RNNs, CNNs, Diffusion, State-Space, MoEs, Hybrids) and techniques (pruning, quantization, novel training methods). Facilitate integration and adaptation of third-party models and research ideas.
*   **Robustness & Maintainability:** Build a stable, well-tested, and clearly documented codebase. Employ validation to ensure configuration integrity.
*   **Usability:** Provide intuitive interfaces (currently a Typer CLI) and clear workflows for standard ML tasks.
*   **Performance & Efficiency:** Optimize for speed and efficient resource use (RAM/VRAM), ensuring viability across different hardware scales, including lower-spec machines.
*   **Adaptability:** While leveraging current best practices and tools, maintain an architectural design that anticipates and can adapt to future shifts in the AI landscape and tooling. <!-- AI Note: Addresses implementation-neutrality goal -->

---

## 2. Core Principles & Current Technology Choices

<!-- AI Note: Reframed to emphasize principles first, current tools second -->
This section outlines the fundamental design principles guiding Craft's development and the specific technologies currently chosen to implement them. These choices reflect current needs and the state of the ecosystem but are subject to evolution as the field progresses.

*   **Principle: Configuration-Driven Experimentation:**
    *   **Need:** A systematic way to define, manage, and reproduce complex experiments by modifying parameters rather than core code.
    *   **Current Solution:** [Hydra](https://hydra.cc/) is used for its powerful configuration composition from YAML files (`conf/`) and command-line override capabilities. See `conf/README.md`.

*   **Principle: Robust Configuration & Validation:**
    *   **Need:** To ensure that configurations are valid *before* launching potentially resource-intensive runs, catching errors early and improving reliability.
    *   **Current Solution:** [Pydantic](https://docs.pydantic.dev/) schemas (`src/craft/config/schemas.py`) are used to define the expected structure and types for configuration files loaded by Hydra, providing validation and clear error reporting.

*   **Principle: Modular & Pluggable Architecture:**
    *   **Need:** To easily swap components like models, datasets, optimizers, and callbacks to facilitate experimentation.
    *   **Current Solution:** Components are typically implemented as Python classes, instantiated dynamically based on configuration (`_target_` key in Hydra configs). Base classes and interfaces define expected contracts.

*   **Principle: High-Performance Computation:**
    *   **Need:** A flexible and efficient backend for numerical computation and automatic differentiation, well-suited for deep learning research and development.
    *   **Current Solution:** [PyTorch](https://pytorch.org/) is the primary deep learning library, chosen for its Pythonic interface, dynamic graphs, strong community support, and extensive ecosystem.

*   **Principle: Accessible & Maintainable Interfaces:**
    *   **Need:** A user-friendly way to interact with the framework's core functionalities (training, generation, etc.) and maintainable code for these interactions.
    *   **Current Solution:** [Typer](https://typer.tiangolo.com/) is used to build the command-line interface (`src/craft/cli/run.py`) due to its simplicity and integration with Python type hints.

*   **Principle: Code Quality & Type Safety:**
    *   **Need:** To maintain a readable, reliable, and maintainable codebase, catching potential errors statically.
    *   **Current Solution:** [Mypy](http://mypy-lang.org/) is used for static type checking, enforcing type hints throughout the `src/` and `tests/` directories.

*   **Guiding Principles (Development Process):**
    *   **Workflow-Driven Development:** Prioritize building and refining functional end-to-end workflows (e.g., data prep -> train -> generate).
    *   **Adherence to Conventions:** Follow established Python/ML best practices where applicable. Deviate thoughtfully when necessary for core goals.
    *   **Single Source of Truth (SSOT):** Strive for SSOT in documentation. This README is the high-level entry point. Component-specific details belong near the component.

---

## 3. Architecture Overview

*(This section remains largely descriptive of the current structure)*

*   **`src/craft/`**: Core library code.
    *   `config/schemas.py`: Pydantic configuration schemas.
    *   `models/`: Model implementations.
    *   `data/`: Data loading, tokenization.
    *   `training/`: Training loop, evaluation, checkpointing, callbacks, etc.
    *   `cli/`: Command-line interface logic.
    *   `utils/`: Common helper functions.
*   **`conf/`**: Hydra YAML configuration files. See `conf/README.md`.
*   **`scripts/`**: Auxiliary/utility scripts. Core tasks use the CLI. See `scripts/README.md`.
*   **`tests/`**: Pytest unit and integration tests. See Section 7 regarding test config locations.
*   **`outputs/`**: Default location for run artifacts (logs, checkpoints), organized by Hydra.
*   **`data/`**: (Optional, top-level) Storage for raw/processed datasets (not tracked by Git).

---

## 4. Key Workflows & Commands

*(Descriptive examples of current usage)*

<!-- TODO: Is this ideal? Would it be better to run with commands like `craft train experiment=<experiment_name>`? -->

*   **Data Preparation:**
    ```bash
    python -m craft.cli.run data prepare ...
    python -m craft.cli.run data train-tokenizer ...
    ```
*   **Training:**
    ```bash
    python -m craft.cli.run train experiment=<experiment_name> ...
    ```
*   **Generation/Inference:**
    ```bash
    python -m craft.cli.run generate text --checkpoint <path/to/checkpoint.pt> ...
    ```
*   **Viewing Logs:**
    ```bash
    tensorboard --logdir outputs/  # (If TensorBoardLogger callback is used)
    ```

---

## 5. Development Status & Tasks

**(AI Agent: Keep 'Current Task' and 'Recent Progress' updated!)**

### Current Task

<!-- TODO: What if we want to switch tasks temporarily, liking taking a detour? Or what happens when we're doing a task and encounter a bug? Then we need to, again, detour into a debugging session, then return to the task. Would this system benefit from something like breadcrumbs to track these detours and help us return to the original task? And when we detour, how do we switch context? -->

*   <!-- AI Note: Fill this with the *very next* specific action item, likely from a low-level plan -->
*   **(Priority: High)** Run all tests in `tests/training/checkpointing/` to check for regressions after fixing strict config validation.

### Recent Progress

<!-- TODO: How much recent progress is too much? Delete? Save to log? New items should consistently be saved to the bottom for chronological ordering. Timestamps would be ideal, but Cursor seems to have a little trouble with them. It can *mostly* get dates right, though it fails to account for time zones. -->

*   <!-- AI Note: List recent completed items here -->
*   Fixed `CheckpointManager` instantiation `TypeError` in multiple tests within `tests/training/checkpointing/test_checkpoint_manager_init_manage.py` by providing required arguments (`experiment_name`) and using fixtures.
*   Refactored `ProgressTracker` (`src/craft/training/progress.py`) for consistency, renaming `tokens_per_second` to `steps_per_second` and `_progress_bar` to `_pbar`.
*   Addressed several test failures in `tests/training/test_generation.py` by:
    *   Simplifying assertions in `test_generate_text_special_token_inference` related to effective token ID logging.
    *   Correcting assertions in `test_generate_text_char_encoding_fallback` regarding `skip_special_tokens`.
    *   Removing obsolete `test_generate_text_decode_type_error_fallback`.
    *   Adapting `test_generate_text_decode_exception_fallback` to test current exception handling logic.
*   Resolved test failures in `tests/training/test_sampling.py` (`test_generate_manual_sampling_temperature`, `test_generate_manual_sampling_top_k`) by updating assertions related to variance checking and expected log messages.
*   Fixed the `Trainer` initialization logic in `tests/training/integration/test_integration_lifecycle.py` by removing the non-existent `trainer.setup()` call and adding assertions for initialized components. The previous edit attempt failed due to syntax errors.
*   Fixed `--split-ratios` argument parsing in the `data prepare` CLI command (`src/craft/cli/dataset_commands.py`).
*   Resolved test failures in `tests/training/trainer/test_trainer_resume.py` related to checkpoint resuming logic and mock assertions.
*   Completed core refactoring of `Trainer` and `TrainingLoop` for `TrainingConfig` integration, dependency simplification, and correct handling of grad accum, AMP, callbacks, etc.
*   Revised and fixed unit tests for `TrainingLoop` (`tests/training/training_loop/`).
*   Successfully refactored and fixed `tests/training/trainer/test_trainer_init_optionals.py` by removing initializer patches and using type assertions.
*   Implemented and tested `data prepare` CLI command (`char`, `subword`) and `PickledDataset` metadata handling.
*   Solidified model configuration process (Pydantic schemas + Hydra configs).
*   Refactored sample generation into `SampleGenerationCallback`.
*   Integrated `Evaluator` into `Trainer`, added perplexity calculation, added tests.
*   Added integration tests for checkpoint/resume scenarios.
*   Extracted component initialization from `Trainer.__init__` to helper functions in `src/craft/training/initialization.py`. This simplifies the `Trainer` constructor. **Note:** The `Trainer` no longer has a `setup()` method; all setup happens within `__init__`.
*   Investigated checkpoint loading (`CheckpointManager.load_checkpoint`) and determined that the `TrainingState` object itself does not perform strict validation on the nested `config` dictionary due to its `extra='allow'` setting and `config: Dict[str, Any]` typing. The `ValidationError` likely occurs later when the loaded config is used, e.g., in `Trainer._resume_from_checkpoint`.
*   Added strict configuration validation logic within `CheckpointManager.load_checkpoint` (`src/craft/training/checkpointing.py`) to compare the loaded checkpoint config against the current model's Pydantic schema, raising `CheckpointLoadError` on mismatch.
*   Refactored error handling in `CheckpointManager.load_checkpoint` to correctly propagate `CheckpointLoadError` from config validation while handling other load errors appropriately.
*   Fixed fixtures (`mock_objects_for_cm`, `checkpoint_manager`) and adapted test logic (`test_load_checkpoint_strict_config_fails`) in `tests/training/checkpointing/test_checkpoint_manager_load.py` to accurately test and verify the strict configuration validation scenario.
*   Implemented VRAM/Throughput logging via TensorBoard callback.

### Near-Term Roadmap / Backlog

*   Stabilize Core (Finish test fixes).
*   Expand End-to-End Testing.
*   Enhance Evaluation Capabilities.
*   Implement Distributed Training support.
*   Improve Documentation & Address Code TODOs.

### Refactoring Roadmap

*(Goal: Systematically align the codebase with README principles/goals, feature by feature. Ensure that all associated tests are aligned to feature implementations.)*

1.  [ ] **Model Interface & Configuration (`src/craft/models/`, `conf/model/`, relevant schemas):**
    *   **Goal:** Enhance versatility (support diverse architectures easily), ensure robust configuration via schemas, clarify `BaseModel` contract, improve adaptability.
2.  [ ] **Data Processing Pipeline (`src/craft/data/`, `conf/data/`, relevant schemas, `cli/dataset_commands.py`):**
    *   **Goal:** Improve modularity (swappable tokenizers/datasets), streamline configuration, enhance adaptability for different data types/sources, ensure clear separation from model logic.
3.  [ ] **Training Loop & Trainer (`src/craft/training/trainer.py`, `training_loop.py`, `callbacks/`, `evaluation.py`, `checkpointing.py`):**
    *   **Goal:** Increase modularity (extract distinct responsibilities like evaluation, checkpointing, device placement), improve callback system flexibility, clarify state management, ensure configuration-driven setup.
4.  [ ] **Configuration System & Schemas (`src/craft/config/schemas.py`, `conf/`, Hydra integration):**
    *   **Goal:** Ensure consistency across features, evaluate deeper Hydra integration (Structured Configs), maintain robustness via Pydantic validation, improve overall clarity.
5.  [ ] **Testing Framework & Coverage (`tests/`):**
    *   **Goal:** Standardize test structure (unit, integration, workflow), improve coverage across all features, ensure tests align with refactored components, consolidate test configurations (addressing Section 7 point).
6.  [ ] **CLI & User Interface (`src/craft/cli/`):**
    *   **Goal:** Ensure clarity, consistency across commands, robust argument parsing/validation, ease of use for core workflows.

### Future Ideas / Experiments

*   ChatGOT Experiments.
    *   Train 1M, 10M, and 100M models using both char and subword. (Review this 6-model design with the AI Agent to see if there's a better design.) Analyze and compare. What improvements do we see as we scale? Or what falls apart as we downsize?
*   Support for New Architectures (RNN, Diffusion, SSM, MoE...).
*   MNIST Support (Multimodality).
*   Third-Party Model Integration (e.g., DeepSeek).
*   Advanced Features (Pruning, Quantization).

---

## 6. Development Standards & Guidelines

*(Prescriptive rules for contributing)*

*   **Style:** PEP 8. Use `black`, `isort` if configured.
*   **Type Checking:** Run `mypy src tests`. Fix errors before commit.
*   **Testing:** Comprehensive coverage (unit, integration, workflow). Use `pytest`. Write tests with code. Keep tests passing.
*   **Documentation:** SSOT. NumPy-style docstrings. Keep READMEs updated.
*   **Dependencies:** Add judiciously per guidelines. Document rationale.
*   **Artifacts:** Use `outputs/`. Ensure checkpoints are complete for resuming.

---

## 7. Review & Refinement Areas

*(Items for future discussion and decision)*

*   **Configuration System:**
    *   Re-evaluate Hydra Structured Configs vs. current Pydantic schema approach.
    *   Review `conf/` organization for clarity.
*   **Testing Strategy:**
    *   Define role/need for smoke tests.
    *   **Standardize Test Config Location:** <!-- AI Note: Proposing a standard --> Recommend using `tests/fixtures/configs/` for minimal, component-specific test configs. Reserve `conf/experiment/test_*.yaml` for full integration tests *requiring* a complex, experiment-like setup. Document this decision once confirmed. This can be revisited during any broader project structure review.
    *   Review use of `tests/assets/`. Define purpose or remove if unused.
    *   Plan for improving test coverage systematically.
*   **Documentation & Knowledge Management:**
    *   Evaluate effectiveness of this README structure long-term.
    *   Consider moving detailed roadmaps/long-term plans to a separate `ROADMAP.md` or project management tool, linking from here.
*   **Codebase & Architecture:**
    *   Address TODOs/FIXMEs.
    *   Periodically review core abstractions for clarity, effectiveness, and alignment with the goal of implementation-neutrality where practical.
    *   Plan a potential broader project structure review.