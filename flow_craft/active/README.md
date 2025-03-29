# ChatGoT Active Flow

This directory contains the active development scripts for the ChatGoT project, focusing on training, evaluation, and sample generation.

## Sample Generation

The sample generation functionality is a critical feature of the ChatGoT system. It allows users to:

1. Generate text samples from trained models
2. Compare outputs between different model checkpoints
3. Debug tokenization and character mapping issues
4. Evaluate model improvements over time

### Key Components

- **generate_simple.py**: Standalone script for generating text from model checkpoints without using the CLI framework
- **generate_batches.ps1**: PowerShell script for batch generation from multiple models and prompts

### Usage

#### Simple Generation

To generate text from a single model checkpoint:

```bash
python flow/active/generate_simple.py --model outputs/models/your_model_checkpoint.pt --prompts "Your prompt here" --max-length 150
```

#### Batch Generation

To generate samples from multiple models and prompts:

```powershell
./flow/active/generate_batches.ps1
```

The batch script will:
1. Find all model checkpoints in the outputs/models directory
2. Use a predefined set of prompts
3. Generate samples for each combination
4. Save outputs to the outputs/samples directory