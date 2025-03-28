# Project Reminders

This document contains important reminders for maintaining the ChatGoT project.

## Documentation

### 📝 Update Documentation When Changing Structure

**IMPORTANT:** Always update documentation when changing the project's directory structure. This includes:

- Update `docs/architecture.md#directory-structure` immediately
- Update the project structure section in the main README.md
- Update any affected README files in subdirectories
- Update import paths in code examples if applicable

See: [Documentation Guidelines](documentation_guidelines.md) for more details.

### 📊 Document Experimental Results

Remember to document all significant experimental results in:

- `flow/performance/` for performance-related findings
- `flow/reference/` for reference materials
- Include raw data in `outputs/evaluation/` directory

## Code Quality

### 🧪 Write Tests for New Code

All new code should have corresponding tests in the `tests/` directory.

### 📋 Follow Style Guidelines

Ensure all code follows the project's style guidelines:

- Use docstrings for all functions, classes, and modules
- Follow PEP 8 conventions
- Use type hints whenever possible

## Git Practices

### 🔀 Keep Commits Focused

Make small, focused commits with clear commit messages.

### 🏷️ Use Semantic Versioning

Follow semantic versioning when creating releases:

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## Project Organization

### 📂 Outputs Directory

Remember that all generated files should go in the appropriate subdirectory of `outputs/`:

- Model checkpoints → `outputs/models/`
- Generated samples → `outputs/samples/`
- Logs → `outputs/logs/`
- Evaluation results → `outputs/evaluation/`
- Visualizations → `outputs/visualizations/`
- Benchmarks → `outputs/benchmarks/`

### 🔍 Debugging

Follow the [Debugging Strategy](debugging_strategy.md) document when debugging issues:

- Create a dedicated debug folder
- Document your debugging process
- Clean up debugging artifacts when done 