# Post-Mortem Rule for Debugging Sessions

This document establishes a protocol for conducting post-mortems after significant debugging sessions and storing the knowledge gained in Flow.

## When to Conduct a Post-Mortem

A post-mortem should be conducted after:

1. Complex debugging sessions that took multiple iterations
2. Resolving non-obvious bugs or issues
3. Fixing infrastructure or environment-related problems
4. Any time the user asks to "remember" what was learned
5. When solving a problem that might recur in the future

## Post-Mortem Framework

### 1. Problem Analysis

- Clearly state what the problem was
- Identify root causes, not just symptoms
- Note which parts were most difficult to diagnose

### 2. Solution Documentation

- Document the specific solutions applied
- Include code snippets when relevant
- Explain why the solution works

### 3. Generalization

- Extract patterns that can be applied to similar problems
- Create reusable tools/utilities if appropriate
- Identify what made this problem unique

### 4. Future Prevention

- Document how to prevent similar issues
- Create checklists or validation steps
- Suggest code or process improvements

## Storage in Flow

Always store post-mortem knowledge in a structured way within Flow:

1. **For specific technical domains**:
   - Create or update documentation in `flow/meta/troubleshooting_guides/{domain}.md`
   - Examples: model_loading.md, deployment.md, data_processing.md

2. **For process improvements**:
   - Store in `flow/meta/processes/{process_name}.md`
   - Examples: code_review.md, testing.md, refactoring.md

3. **For architectural decisions**:
   - Document in `flow/meta/architecture/{component}.md`
   - Examples: model_architecture.md, data_pipeline.md

## Trigger Words

Whenever the user uses any of these phrases, it should trigger the post-mortem process:

- "remember this"
- "remember what we learned"
- "store this for later"
- "save this knowledge"
- "what did we learn"
- "post-mortem"
- "document this"

## Example Post-Mortem Structure

```
# Post-Mortem: [Issue Title]

## Problem
[Clear description of what went wrong]

## Root Causes
1. [Primary cause]
2. [Secondary cause]
3. [Contributing factors]

## Solution
[Description of how the issue was resolved]

## Code Snippets
```python
# Example code that fixes the issue
```

## Lessons Learned
1. [Key takeaway 1]
2. [Key takeaway 2]
3. [Key takeaway 3]

## Future Prevention
- [Step 1 to prevent this issue]
- [Step 2 to prevent this issue]
```

## Automation Opportunities

Consider creating utilities to help with post-mortems:

1. Debugging log analyzers
2. Code diff summarizers
3. Checklist generators for common issues

By following this protocol, we ensure that knowledge gained from debugging sessions is properly documented and accessible for future reference, preventing repeated issues and reducing debugging time. 