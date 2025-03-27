# Debugging Guide

_Version: 1.0 | Last Updated: 2025-03-26_

## Overview

This guide documents the debugging workflow, tools, and resources for the AI model development environment. It provides structured instructions for identifying and resolving issues during model development.

## Related Documentation

- [Debug Log](../logs/debug_log.md) - Technical implementation details and code changes
- [Progress Log](../logs/progress_log.md) - Current status and next steps
- [Implementation Guide](implementation_guide.md) - Prioritized roadmap for features

## Key Resources

### Progress Tracking
The [Progress Log](../logs/progress_log.md) provides a high-level checklist showing:
- Which issues have been fixed
- Features that need to be re-enabled
- Current model configurations
- Session notes

### Technical Documentation
The [Debug Log](../logs/debug_log.md) maintains technical details including:
- Specific code changes and implementations
- Configuration modifications
- Memory optimization strategies
- Current behavior and issues

### Implementation Planning
The [Implementation Guide](implementation_guide.md) contains:
- Prioritized implementation steps
- Task breakdowns with checkboxes
- Code examples for key features
- Testing methodology

## Command Reference

### Running Training with Clean Configuration
```bash
# Launch training with default parameters
python scripts/train_with_samples.py configs/models/debug/chatgot_small_14M_clean.yaml

# Launch with specific arguments
python scripts/train_with_samples.py \
    configs/models/debug/chatgot_small_14M_clean.yaml \
    --batch_size 16 \
    --devices 1
```

### Resuming from Checkpoint
```bash
# Resume training from a specific checkpoint
python scripts/train_with_samples.py \
    configs/models/debug/chatgot_small_14M_clean.yaml \
    --resume_from models/chatgot_small_14M_clean_[TIMESTAMP]_step_[STEP].pt
```

## Debugging Workflow

1. Check the [Progress Log](../logs/progress_log.md) to understand what's been done and what's next
2. Implement the highest priority item from the [Implementation Guide](implementation_guide.md)
3. Test thoroughly using the suggested testing methodology
4. Document any issues or solutions in the [Debug Log](../logs/debug_log.md)
5. Update all tracking files with your progress
6. Commit changes with clear messages referencing the feature/fix

## Configuration Files

| Filename | Purpose | Parameters |
|----------|---------|------------|
| chatgot_test.yaml | Original model | 85M |
| chatgot_small_25M.yaml | Smaller model | 25M |
| chatgot_small_25M_memory.yaml | Memory-optimized version | 25M |
| chatgot_small_14M_clean.yaml | Clean debugging version | 14M |

## Next Session Guide

If starting a new session:

1. Read the [Progress Log](../logs/progress_log.md) to understand current status
2. Check the [Debug Log](../logs/debug_log.md) for technical details of recent changes
3. Follow next steps from the [Implementation Guide](implementation_guide.md)
4. Update all documentation files after completing tasks
5. Record your activities in the [Working Memory](../../working_memory.md) file 