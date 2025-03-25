# Project Organization

## Directory Structure

```
.
├── configs/                    # Configuration files
│   ├── models/                # Model configurations
│   │   ├── chatgot_small_char.yaml  # Character-level model based on GPT-2 small
│   │   └── gpt2_small.yaml   # Standard GPT-2 small model
│   ├── data/                  # Data processing configurations
│   └── experiments/           # Experiment configurations
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data files
├── docs/                      # Documentation
│   ├── organization.md       # This file
│   └── experiments/          # Experiment documentation
├── scripts/                   # Utility scripts
│   ├── chatgot              # CLI entry point
│   ├── continue_pipeline.py  # Pipeline continuation script
│   └── train_with_samples.py # Training script with sample generation
├── src/                      # Source code
│   ├── cli/                 # CLI implementation
│   ├── data/                # Data handling
│   ├── experiments/         # Experiment framework
│   ├── models/              # Model implementations
│   ├── training/            # Training utilities
│   └── utils/               # Utility functions
├── tests/                   # Test files
├── .github/                 # GitHub configuration
├── pyproject.toml          # Project configuration
└── README.md               # Project documentation
```

## Key Components

### Models
- `chatgot_small_char`: Character-level language model based on GPT-2 small architecture
- `gpt2_small`: Standard GPT-2 small model with subword tokenization

### Configuration Files
- Model configurations in `configs/models/`
- Data processing configurations in `configs/data/`
- Experiment configurations in `configs/experiments/`

### Training
- Main training script: `scripts/train_with_samples.py`
- CLI interface: `scripts/chatgot`
- Pipeline continuation: `scripts/continue_pipeline.py`

### Data Processing
- Data loading and preprocessing in `src/data/`
- Tokenization and dataset creation utilities

### Experiment Framework
- Experiment runner in `src/experiments/`
- MLflow integration for tracking
- Resource monitoring

## Development Guidelines

1. **Configuration Management**
   - Use YAML files in `configs/` for all configurations
   - Follow the established naming convention for model configs
   - Document all configuration parameters

2. **Code Organization**
   - Keep related functionality together
   - Use clear, descriptive names
   - Follow Python best practices

3. **Testing**
   - Write tests for new features
   - Maintain test coverage
   - Use pytest for testing

4. **Documentation**
   - Update docs when making changes
   - Document new models and configurations
   - Keep experiment documentation current 