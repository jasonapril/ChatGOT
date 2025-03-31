# Troubleshooting and Debugging Guide

This document consolidates strategies, frameworks, and processes for effectively troubleshooting and debugging issues within the project, derived from various sources within the Flow system.

## 1. General Troubleshooting Framework

*Based on `general_troubleshooting_framework.md`*

### Core Principles of Effective Troubleshooting

#### 1.1 Systematic Isolation
- **Divide and Conquer**: Break the problem into smaller, testable components
- **Control Variables**: Change one thing at a time
- **Establish Baselines**: Create minimal working examples to compare against
- **Log Boundaries**: Clearly mark the transitions between components

#### 1.2 Evidence-Based Investigation
- **Follow the Warnings**: Pay close attention to warning messages - they often contain the exact information needed
- **Trust the Errors**: Error messages typically point directly to the problem area
- **Capture Relevant Metrics**: Measure before and after each change
- **Preserve Error States**: Save logs and error outputs for analysis

#### 1.3 Incremental Resolution
- **Start Simple**: Try the simplest solution first
- **Verify Each Step**: Test after each change to isolate effects
- **Rollback Capability**: Be able to undo changes that don't help
- **Document Progress**: Keep track of what's been tried and the results

#### 1.4 Pattern Recognition
- **Compare Known Good States**: Identify differences between working and non-working states
- **Look for Name Mismatches**: Different naming conventions often cause compatibility issues
- **Dimension/Size Inconsistencies**: Many errors stem from mismatched sizes or capacities
- **Configuration Conflicts**: Check for conflicting settings across components

### Structured Debugging Process

#### Phase 1: Problem Definition
1. **Document Symptoms**
   - What exactly is happening?
   - What should be happening instead?
   - Is the behavior consistent or intermittent?
2. **Gather Context**
   - When did the problem start?
   - What changed recently?
   - Are there dependencies involved?
3. **Classify the Problem**
   - Is it a configuration issue?
   - Is it a compatibility problem?
   - Is it a resource constraint?
   - Is it a logical error?

#### Phase 2: Investigation
1. **Create a Minimal Test Case**
   - Remove unnecessary components
   - Simplify the environment
   - Use mock objects if appropriate
2. **Inspect the Boundaries**
   - Check interfaces between components
   - Verify data formats at handoff points
   - Test components in isolation
3. **Analyze System Logs**
   - Look for warning messages
   - Check timestamps around failures
   - Note any resource constraints
4. **Compare with Working States**
   - Diff configurations
   - Compare version numbers
   - Check for environment differences

#### Phase 3: Resolution
1. **Hypothesis Formation**
   - Based on evidence, form a clear hypothesis
   - The hypothesis should be testable
   - It should explain the observed symptoms
2. **Targeted Testing**
   - Design a specific test for your hypothesis
   - Make minimal changes to test
   - Document expected outcomes before testing
3. **Incremental Fixes**
   - Implement the simplest fix first
   - Test after each change
   - Revert if a change doesn't help
4. **Verify Full Solution**
   - Test the entire system
   - Verify edge cases
   - Ensure no regression in other areas

#### Phase 4: Knowledge Capture (See also Section 3: Post-Mortem Process)
1. **Document Root Causes**
   - What was the fundamental issue?
   - Why did it manifest in this particular way?
   - Were there multiple contributing factors?
2. **Generalize Lessons**
   - What pattern does this problem follow?
   - Could this affect other systems?
   - What general principle can be extracted?
3. **Create Prevention Mechanisms**
   - Update validation processes
   - Add automated tests
   - Improve error messaging
   - Create documentation

### Common Troubleshooting Patterns

#### Interface Mismatches
**Signs**: "Not found"/"missing" errors, components can't communicate, data transformations fail.
**Approach**: Check names, verify types/formats, test isolation, add mapping.

#### Resource Constraints
**Signs**: Timeouts, performance degradation, crashes under load.
**Approach**: Monitor usage, check leaks, test reduced load, optimize paths.

#### Configuration Drift
**Signs**: "Works on my machine", environment-specific failures, intermittent issues.
**Approach**: Compare configs, check env vars, verify versions, create reproducible environments.

#### Data Inconsistencies
**Signs**: Validation errors, unexpected outputs, processing failures.
**Approach**: Validate input, check logic, trace flow, create integrity checks.

### Tools for Effective Troubleshooting
- **Logging Framework**: Structured, leveled logs with context and timestamps.
- **Diff Tools**: Compare configs, code changes.
- **Monitoring**: Resource usage, performance metrics, error rates.
- **Testing Harnesses**: Isolated tests, reproducible cases, regression detection.

### General Lessons Learned
- **Start Simple**: Check fundamentals first; complex issues often have simple causes.
- **Follow Warnings & Errors**: Don't ignore minor warnings; trust error message locations.
- **Trust but Verify**: Assume components work as documented, but verify behavior in context.
- **Document As You Go**: Record steps, observations, and state during the process.
- **Use Explicit Configuration**: Don't rely on default parameters; extract and use configuration from known sources (like checkpoints) when possible.
- **Handle Interfaces Robustly**: Implement robust handling (e.g., key mapping, validation) for mismatches at component boundaries.
- **Separate Concerns**: Test components (like data vs. model processing) independently.
- **Log Thoroughly**: Ensure logging captures configuration, dimensions, boundaries, and context around errors.

---

## 2. Debugging Strategy (for Code-Level Issues)

*Based on `debugging_strategy.md`*

### 2.1 Setting Up a Debugging Environment

#### Create a Dedicated Debug Folder
Use a dedicated, temporary folder (e.g., within `scratch/` or a top-level `debug/`) for experimental code. Name it descriptively (e.g., `debug/YYYYMMDD_issue_description/`).

#### Establish a Debug Log
Maintain a log file (e.g., `debug_log.md`) within the debug folder, documenting: issue, hypotheses, attempts, results, errors.

### 2.2 Organizing Debug Code

#### Naming Conventions
Use clear names: `debug_[component].py`, `test_[hypothesis].py`, `fix_attempt_[number].py`.

#### Separation of Concerns
Separate scripts for diagnostics, hypothesis testing, fix attempts, and evaluation.

### 2.3 Documentation During Debugging

#### Inline Documentation
Use docstrings/headers in debug files explaining purpose, hypothesis, expected outcome.

#### Commenting Test Results
Add comments directly in debug code documenting results (`# RESULT: Failed`, `# RESULT: Success`).

### 2.4 From Debug to Solution

#### Solution Extraction Process
1. Extract core solution code.
2. Create a clean implementation (separate from debug code).
3. Test clean implementation independently.
4. Document solution clearly (see Post-Mortem Process below).
5. Integrate solution into the main codebase.

### 2.5 Cleanup Protocol

#### Cleanup Checklist
After resolving:
- [ ] Integrate successful solution into codebase.
- [ ] Document solution/insights (see Post-Mortem Process below).
- [ ] Create automated tests for regression prevention.
- [ ] Archive useful debug artifacts (if any).
- [ ] Remove the dedicated debug folder and its contents.

#### Cleanup Script (Optional)
A simple script can help remove the debug directory.

---

## 3. Post-Mortem Process

*Based on `post_mortem_rule.md`*

### 3.1 When to Conduct a Post-Mortem

Conduct after:
- Complex/multi-iteration debugging.
- Resolving non-obvious bugs.
- Fixing infrastructure/environment issues.
- User requests knowledge capture ("remember this", "document this", etc.).
- Solving potentially recurring problems.

### 3.2 Post-Mortem Framework

1. **Problem Analysis**: State problem, identify root causes, note difficulties.
2. **Solution Documentation**: Document specific solutions, include code, explain why it works.
3. **Generalization**: Extract patterns, create reusable tools, note unique aspects.
4. **Future Prevention**: Document prevention steps, suggest checklists/validation, propose improvements.

### 3.3 Storage in Flow

Store post-mortem knowledge systematically:
- **Specific Technical Guides**: Update or create files in `flow/meta/troubleshooting_guides/` (e.g., `model_loading.md`). See Section 4.
- **Process Improvements**: Store in relevant process documents (e.g., within `flow/meta/` or `flow/system/`).
- **Architectural Decisions**: Document in relevant architecture documents.

### 3.4 Trigger Words for Post-Mortem

Initiate process if user uses phrases like: "remember this", "document this", "what did we learn", "post-mortem", "save this knowledge".

### 3.5 Example Post-Mortem Structure

```markdown
# Post-Mortem: [Issue Title]

## Problem
[Clear description of what went wrong]

## Root Causes
1. [Primary cause]
2. [Secondary cause]
3. [Contributing factors]

## Solution
[Description of how the issue was resolved]

## Code Snippets (Optional)
\`\`\`python
# Example code that fixes the issue
\`\`\`

## Lessons Learned
1. [Key takeaway 1]
2. [Key takeaway 2]

## Future Prevention
- [Step 1 to prevent this issue]
- [Step 2 to prevent this issue]
``` 