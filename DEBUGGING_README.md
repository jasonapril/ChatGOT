# ChatGoT Debugging Guide

This directory contains files to track our debugging progress and implementation plan for the ChatGoT model training system.

## Quick Start
- Check **DEBUGGING_PROGRESS.md** for current status and next steps
- See **DEBUG_LOG.md** for technical implementation details
- Follow **IMPLEMENTATION_PLAN.md** for the prioritized roadmap

## Key Files

### DEBUGGING_PROGRESS.md
A high-level checklist showing:
- What issues have been fixed
- What features need to be re-enabled
- Current model configurations
- Session notes

### DEBUG_LOG.md
Technical documentation with:
- Specific code changes and implementations
- Configuration modifications
- Memory optimization strategies
- Current behavior and issues

### IMPLEMENTATION_PLAN.md
Detailed roadmap with:
- Prioritized implementation steps
- Task breakdowns with checkboxes
- Code examples for key features
- Testing methodology

## Command Reference

### Running Training with Clean Configuration
```bash
python scripts/train_with_samples.py configs/models/debug/chatgot_small_14M_clean.yaml
```

### Resuming from Checkpoint
```bash
python scripts/train_with_samples.py configs/models/debug/chatgot_small_14M_clean.yaml --resume_from models/chatgot_small_14M_clean_[TIMESTAMP]_step_[STEP].pt
```

## Debugging Workflow

1. Check **DEBUGGING_PROGRESS.md** to understand what's been done and what's next
2. Implement the highest priority item from **IMPLEMENTATION_PLAN.md**
3. Test thoroughly using the suggested testing methodology
4. Document any issues or solutions in **DEBUG_LOG.md**
5. Update all tracking files with your progress
6. Commit changes with clear messages referencing the feature/fix

## Configuration Files

- **chatgot_test.yaml**: Original 85M parameter model
- **chatgot_small_25M.yaml**: Smaller 25M parameter model
- **chatgot_small_25M_memory.yaml**: Memory-optimized version
- **chatgot_small_14M_clean.yaml**: Clean debugging version (14M)

## Next Session Guide

If starting a new session:
1. Read **DEBUGGING_PROGRESS.md** to understand current status
2. Check **DEBUG_LOG.md** for technical details of recent changes
3. Follow next steps from **IMPLEMENTATION_PLAN.md**
4. Update all documentation files after completing tasks 