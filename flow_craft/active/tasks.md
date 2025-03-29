# Tasks

*Part of the Flow System. See also: [Guidelines](../system/guidelines.md), [Improvements](../planning/improvements.md).*

This file serves as the working memory for all active tasks in the project. It's designed to provide at-a-glance visibility into what's currently being worked on, what's coming up next, and what has recently been completed.

## Active Tasks

- ### 🔴 Complete Model Architecture Refactoring
  - Create proper abstraction hierarchy for model types
  - Implement clean interfaces between model components
  - Separate model definition from training logic
  - Ensure consistent naming conventions
  - 🔍 Planned (See [Refactoring Plan](../planning/refactoring_plan.md))

- ### 🟠 Standardize Data Pipeline
  - Create a unified data loading interface
  - Standardize preprocessing steps
  - Implement dataset versioning
  - Add data validation hooks
  - 🔍 Planned (See [Refactoring Plan](../planning/refactoring_plan.md))

- ### 🟡 Implement Basic Unit Tests
  - Create unit tests for the CLI functionality
  - Add tests for the utility modules (especially `checkpoint.py` and `io.py`)
  - Configure GitHub Actions for running tests automatically
  - ✅ Completed 2025-03-26 (See log entry)

## Upcoming Tasks

- ### Review Dependencies
  - Which libraries does Craft depend on? Are they all justified?

- ### 🟠 Update Configuration System
  - Replace hardcoded values with configuration options
  - Implement validation for configuration values
  - Create a configuration management system
  - 🔍 Planned (See [Refactoring Plan](../planning/refactoring_plan.md))

- ### 🟡 Refactor Training Loop
  - Extract common training functionality into a base trainer
  - Implement hooks for custom training behavior
  - Add support for different optimization strategies
  - 🔍 Planned (See [Refactoring Plan](../planning/refactoring_plan.md))

- ### 🟡 Set up CI/CD pipeline
  - Configure GitHub Actions for automated testing
  - Set up documentation generation and deployment
  - Implement version management
  - 🔍 Planned

## Completed Tasks

- ### 🟠 Update project entry point
  - Update `pyproject.toml` to use the new Typer-based CLI
  - Ensure the correct entry point script is used (`src.cli.run:app`)
  - Remove any remaining references to the old CLI
  - ✅ Completed 2025-03-26 (See log entry)

- ### 🟡 Complete documentation update
  - Update project README.md with improved installation and usage instructions
  - Create a getting started guide with examples
  - Add architecture overview
  - ✅ Completed 2025-03-26 (See log entry)

- ### 🟠 Implement missing utility modules
  - Create checkpoint.py for model checkpoint functionality
  - Implement io.py for file operations
  - Update imports in the codebase to use new modules
  - ✅ Completed 2025-03-26 (See log entry)

- ### 🟠 Consolidate CLI implementation
  - Move from Click-based to Typer-based CLI
  - Ensure all commands work properly
  - Test different command options
  - ✅ Completed 2025-03-26 (See log entry)

- ### 🟡 Implement Folder Structure
  - Organize code into modular components
  - Create dedicated directories for models, data, training
  - Update imports throughout the codebase
  - ✅ Completed 2025-03-25 (See log entry)

## Update Guidelines

- Add new tasks as they arise, including a brief description and the relevant links to detailed plans.
- When starting a task, insert an entry with a timestamp and clear breadcrumbs.
- Update the active task list frequently; a task is considered active if it appears here.
- Mark tasks with appropriate status indicators:
  - ⏳ In progress
  - ✅ Completed (ready for removal)
  - 🔄 Under review
  - ⏸️ Paused
  - 🚩 Blocked
- Use priority markers as defined in the [Guidelines](../system/guidelines.md).
- Recently completed tasks should remain in the "Completed Tasks" section for a short period (1-2 weeks max) for reference before being archived.
- Archive completed tasks to the logs directory with appropriate date stamps when they're no longer needed for immediate reference.
- Maintain only recent logs (5-10 entries) in the Log section. Older logs are archived to flow/logs/.

## Log

- [2025-03-27] Updated project priorities to focus on completing refactoring before performance optimization. Created a comprehensive refactoring plan with four main focus areas: model architecture, data pipeline, configuration system, and training loop. See [Refactoring Plan](../planning/refactoring_plan.md) for details.
- [2025-03-27] Restructured the Flow system into a more organized directory structure with clear separation of concerns: active for working memory, system for guidelines, planning for future-oriented documents, and reference for implementation details. Updated all cross-references and created README files for each directory to explain their purpose.
- [2025-03-27] Created end-to-end example notebook demonstrating the ChatGoT workflow. The notebook `notebooks/chatgot_workflow.ipynb` shows the complete process from data preparation to model training and text generation, with visualizations of model performance and example outputs.
- [2025-03-27] Created improvement_suggestions.md to document potential enhancements to the Flow system, including automation, visual improvements, integration options, knowledge management features, process refinements, and team collaboration capabilities.
- [2025-03-27] Moved the Prevention Plan from logs/ to the main flow/ directory to emphasize its critical importance for system integrity. Updated all references to reflect the new location.
- [2025-03-27] Restructured flow.md to better emphasize its role as working memory. Removed redundant priority legend, added an explanatory introduction, clarified the purpose of the "Completed Tasks" section as temporary reference before archiving, and linked to system.md as the source of truth for guidelines.
- [2025-03-27] Standardized priority system across flow documentation. Ensured consistent priority indicators between flow.md and system.md files (🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low). Removed redundant priority text in task description to improve readability and reduce maintenance overhead. Created a Prevention Plan to avoid similar inconsistencies in the future.
- [2025-03-26] Implemented unit tests for CLI and utility modules. Created test_checkpoint.py, test_io.py, and test_cli.py test files to validate the functionality of the new utility modules and CLI commands. Added comprehensive tests with proper test fixtures and mocks to ensure robustness. Updated run_all_tests.py to discover and run the new tests.
- [2025-03-26] Updated project entry point in pyproject.toml to use the new Typer-based CLI (src.cli.run:app). Added typer as a dependency and removed references to the old CLI entry point.
- [2025-03-26] Updated project documentation to reflect the consolidated CLI and new utility modules. Modified README.md with clearer installation and usage instructions, and updated the getting_started.md guide with detailed CLI commands.

_Note: Older logs have been archived to the logs directory_

## Next Steps
1. Complete model architecture refactoring
2. Standardize data pipeline interfaces
3. Update configuration system
4. Refactor training loop

## Task List

## Current Tasks

### High Priority
- [ ] **Investigate text generation issues** (In Progress)
  - [x] Create initial sample generation script
  - [x] Document text generation investigation process in Flow
  - [x] Test different generation parameters
  - [ ] **Analyze train_with_samples.py generation process** (Current focus)
    - [ ] Extract character dataset initialization code
    - [ ] Extract model configuration and loading process
    - [ ] Analyze generation function implementation
    - [ ] Identify any warm-up or conditioning steps
    - [ ] Create a direct port of the working generation approach
  - [ ] Compare tokenization between training and standalone scripts
  - [ ] Test with various prompts and model checkpoints

### Medium Priority

### Low Priority

## Completed Tasks
- [x] Create a simple script that can generate text samples (sample_generator.py)
- [x] Document text generation investigation in Flow
- [x] Set up better error handling and verbose logging in generation script

## Resources
- `train_with_samples.py`: Script that successfully generates coherent text samples during training
- `sample_generator.py`: Standalone script for text generation (currently producing incoherent output)
- `text_generation_investigation.md`: Documentation of our investigation process 