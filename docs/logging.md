# Logging and Experiment Tracking

This document describes the logging and experiment tracking conventions for the ChatGoT project.

## Directory Structure

The project uses the following logging directory structure:

```
runs/
├── chatgot_small_char_20250324_134501/    # Run with config name and timestamp
│   ├── events.out.tfevents.*.v2           # TensorBoard event files
│   └── training.log                       # Text log file
└── archive/                               # Old format logs (archived)
```

## Naming Convention

Experiment directories follow this naming pattern:
- `{config_name}_{timestamp}`
  - `config_name`: Name of the configuration file without extension (e.g., `chatgot_small_char`)
  - `timestamp`: Format `YYYYMMDD_HHMMSS` (e.g., `20250324_134501`)

## Log Types

The project utilizes several types of logs:

1. **Console Output**: Displayed in the terminal during training
2. **Text Log File**: Full logs stored in `training.log`
3. **TensorBoard Logs**: 
   - Loss curves
   - Throughput metrics
   - Generated text samples
   - Model hyperparameters

## Log Format

Log messages follow this format:
```
YYYY-MM-DD HH:MM:SS,mmm - LEVEL - Message
```

Example:
```
2025-03-24 14:05:33,642 - INFO - Using device: cuda
```

## Accessing Logs

### Text Logs

Text logs can be viewed with any text editor:
```bash
cat runs/chatgot_small_char_20250324_134501/training.log
```

### TensorBoard Logs

TensorBoard logs can be visualized using:
```bash
tensorboard --logdir=runs/chatgot_small_char_20250324_134501
```

Or to compare multiple runs:
```bash
tensorboard --logdir=runs
```

## Generated Samples

Sample text generated during training is:
1. Logged to the console
2. Saved in the text log file
3. Recorded in TensorBoard under the `samples` tag

## Model Checkpoints

Model checkpoints follow a similar naming convention:
```
models/chatgot_small_char_20250324_134501_epoch_1.pt  # Epoch checkpoint
models/chatgot_small_char_20250324_134501_final.pt    # Final model
``` 