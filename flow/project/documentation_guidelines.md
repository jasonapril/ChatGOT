# Documentation Guidelines

This document outlines the guidelines and best practices for maintaining documentation in the Craft project.

## Key Principles

1. **Documentation as Code**: Treat documentation with the same level of importance as code
2. **Keep in Sync**: Documentation must stay synchronized with the actual code and structure
3. **Anticipate Questions**: Document with future users/contributors in mind
4. **Consistent Format**: Follow consistent style and formatting conventions

## When to Update Documentation

Documentation should be updated whenever:

1. **Adding new features or functionality**
2. **Changing existing APIs or interfaces**
3. **Modifying the directory structure**
4. **Introducing new dependencies**
5. **Changing the build process or workflow**
6. **Fixing bugs that impact behavior described in documentation**

## Directory Structure Documentation

**IMPORTANT: Always update directory structure documentation when changing the folder organization.**

The main directory structure documentation is located at:
- `docs/architecture.md` in the "Directory Structure" section

When adding, moving, or removing directories:

1. Update the directory tree in `docs/architecture.md#directory-structure`
2. Update relevant section descriptions
3. Update any affected README files
4. If the change affects code imports, update import examples

## Types of Documentation

### 1. Code Documentation

- **Docstrings**: All modules, classes, and functions should have docstrings
- **Comments**: Complex code sections should have explanatory comments
- **Type Hints**: Use type hints to document expected parameter and return types

### 2. Directory/Repository Documentation

- **README files**: Each significant directory should have a README.md explaining its purpose
- **Directory structure documentation**: Maintain accurate directory tree in `docs/architecture.md#directory-structure`

### 3. Process Documentation

- **Setup guides**: Instructions for setting up the project
- **Contribution guidelines**: How to contribute to the project
- **Workflow documentation**: How to perform common tasks

## Documentation Style

- Use Markdown for all documentation files
- Code snippets should be fenced with triple backticks and include language identifier
- Use headings to create a hierarchical structure
- Keep line length reasonable (approximately 100 characters)
- Use links to reference other documentation files or external resources

## Documentation Checklist

When making changes, go through this checklist:

- [ ] Updated relevant docstrings
- [ ] Updated README files if needed
- [ ] Updated directory structure documentation if folder structure changed
- [ ] Added examples for new features
- [ ] Verified links still work
- [ ] Checked that documentation renders correctly

## Directory Structure Change Procedure

When modifying the directory structure:

1. Plan the changes and document the plan
2. Make the changes to the directory structure
3. **Immediately update `docs/architecture.md#directory-structure`**
4. Update any affected README files
5. Update any code examples or import paths in documentation
6. Verify the documentation matches the actual structure 