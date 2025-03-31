# Scripts (`scripts/`)

This directory contains standalone utility scripts that support development, analysis, or interaction with the Craft project, but are not part of the core library (`src/`) code.

## Contents:

*   `get_latest_checkpoint.py`: Finds the latest checkpoint file across multiple Hydra run directories based on specified criteria (model, dataset, experiment name) and step count.
*   `check_model_params.py`: Analyzes and compares expected vs. actual parameter counts for a model configuration, useful for verification and debugging.

(Note: This directory was significantly cleaned, removing legacy pipeline scripts, redundant CLI entry points, and utilities whose functionality is now covered by the main `src/cli/run.py` interface or integrated experiment tracking.)

## Usage:

These scripts are typically run directly from the command line, e.g.:
```bash
python scripts/get_latest_checkpoint.py --experiment-name chatgot_25m
python scripts/check_model_params.py --config conf/experiment/chatgot_25m.yaml
``` 