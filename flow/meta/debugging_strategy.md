# Debugging Strategy for Complex Issues

This document outlines a structured approach to debugging complex issues, with a focus on keeping the codebase clean and organized during the process.

## 1. Setting Up a Debugging Environment

### Create a Dedicated Debug Folder

For any significant debugging effort, create a dedicated debug folder to contain all experimental code:

```bash
# Create a debug folder with the current date and a descriptive name
mkdir -p debug/$(date +%Y%m%d)_issue_description
```

Example: `debug/20250327_model_loading_issue/`

### Establish a Debug Log

Create a markdown file to document your debugging process:

```bash
touch debug/$(date +%Y%m%d)_issue_description/debug_log.md
```

The log should include:
- Issue description
- Hypotheses 
- Attempted solutions
- Results of each attempt
- Relevant error messages and logs

## 2. Organizing Debug Code

### Naming Conventions

Use clear, explicit naming for all debug files:

- `debug_[component]_[specific_focus].py`
- `test_[hypothesis].py`
- `fix_attempt_[number].py`

Examples:
- `debug_model_loading.py`
- `test_layer_dimension_hypothesis.py`
- `fix_attempt_1_key_mapping.py`

### Separation of Concerns

Separate different aspects of the debugging process:

1. **Diagnostic Scripts**: Code that only examines the issue
2. **Test Scripts**: Code that tests specific hypotheses
3. **Fix Attempts**: Code that implements potential solutions
4. **Evaluation Scripts**: Code that validates fixes

## 3. Documentation During Debugging

### Inline Documentation

Every debug file should include a header with:

```python
"""
Debug script for [issue description]

Purpose: [What this script is trying to achieve]
Hypothesis: [What you're testing, if applicable]
Expected outcome: [What you expect to see if successful]

Created: YYYY-MM-DD
"""
```

### Commenting Test Results

Add comments with results directly in the debug files:

```python
# RESULT: Failed with error XYZ
# RESULT: Partial success - fixed issue A but not B
# RESULT: Success - confirmed hypothesis
```

## 4. From Debug to Solution

### Solution Extraction Process

Once a fix is found, follow this process to integrate it:

1. **Extract Core Solution**: Identify the essential code changes needed
2. **Clean Implementation**: Create a clean implementation separate from debug code
3. **Test Clean Implementation**: Ensure it works independently of debug context
4. **Document Solution**: Create clear documentation of what was fixed and how
5. **Integration**: Integrate the solution into the main codebase

### Solution Documentation

Create a post-mortem document that includes:

```markdown
# Issue Resolution: [Issue Name]

## Problem
[Clear description of what was wrong]

## Root Causes
[List of root causes identified]

## Solution
[Description of the solution]

## Key Insights
[Important lessons learned]

## Prevention
[How to prevent similar issues]
```

## 5. Cleanup Protocol

### Cleanup Checklist

After resolving the issue:

- [ ] Move successful solution to the appropriate place in the codebase
- [ ] Document the solution and insights in `flow/meta/troubleshooting_guides/`
- [ ] Create automated tests to prevent regression
- [ ] Archive useful debug artifacts (if any)
- [ ] Run cleanup script to remove all debugging code

### Cleanup Script Template

Create a cleanup script that targets only debugging artifacts:

```powershell
# Debug cleanup script
$debugPath = "debug/20250327_issue_name"
Write-Host "Archiving useful debug artifacts..."
# Archive useful files if needed
# ...

Write-Host "Removing debug directory..."
Remove-Item -Path $debugPath -Recurse -Force
```

## 6. Debugging Workflow Example

1. **Issue Identification**:
   - Identify model loading issue with error messages
   - Create `debug/20250327_model_loading_issue/`

2. **Diagnostic Phase**:
   - Create `debug_inspect_checkpoint.py` to examine checkpoint structure
   - Create `debug_model_structure.py` to print model architecture
   - Document findings in `debug_log.md`

3. **Hypothesis Formation**:
   - Form hypothesis about dimension mismatch
   - Create `test_dimensions.py` to verify

4. **Fix Attempts**:
   - Create `fix_attempt_1_dimensions.py` to try dimension fix
   - Create `fix_attempt_2_key_mapping.py` to try key mapping

5. **Solution Integration**:
   - Extract working solution into `scripts/generate_samples.py`
   - Test to ensure it works properly

6. **Cleanup**:
   - Document solution in `flow/meta/troubleshooting_guides/model_loading.md`
   - Run cleanup script to remove all debug directories

By following this structured approach, debugging efforts remain contained, organized, and don't pollute the main codebase. 