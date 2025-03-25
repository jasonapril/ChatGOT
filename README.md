# ChatGoT

An experimental framework for exploring and developing language models and other AI architectures.

## Project Goals

1. **Architecture Exploration**
   - Experiment with different model architectures
   - Compare performance and efficiency
   - Develop novel approaches

2. **Performance Optimization**
   - Memory efficiency
   - Training speed
   - Resource utilization

3. **Novel Applications**
   - Character-level language modeling
   - Efficient attention mechanisms
   - Custom tokenization schemes

4. **Research Platform**
   - Reproducible experiments
   - Comprehensive monitoring
   - Extensible framework

## Documentation

We maintain comprehensive documentation in the `docs/` directory:

- [Documentation Index](docs/INDEX.md) - Central hub for all documentation
- [Debugging Progress](DEBUGGING_PROGRESS.md) - Current debugging status
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Feature implementation roadmap

For debugging and development tracking, see:
- [Debugging README](DEBUGGING_README.md) - Guide to our debugging process
- [Technical Debug Log](DEBUG_LOG.md) - Detailed code changes and fixes

## Project Structure

```
.
├── configs/                    # Configuration files
│   ├── models/                # Model configurations
│   ├── data/                  # Data processing configurations
│   └── experiments/           # Experiment configurations
├── data/                      # Data directory
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── src/                      # Source code
└── tests/                   # Test files
```

## Features

### Model Architectures
- Character-level language models
- Standard transformer architectures
- Custom attention mechanisms
- Efficient training optimizations

### Training and Optimization
- Mixed precision training
- Gradient checkpointing
- Memory-efficient attention
- Custom learning rate schedules

### Development Tools
- Comprehensive CLI
- Experiment tracking
- Resource monitoring
- Performance profiling

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChatGoT.git
   cd ChatGoT
   ```

2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Usage

### Training a Model

```bash
# Train character-level model
chatgot train --config-path configs/models/chatgot_small_char.yaml

# Train standard GPT-2 model
chatgot train --config-path configs/models/gpt2_small.yaml
```

### Generating Text

```bash
chatgot generate --model-path path/to/model --prompt "Your prompt here"
```

### Running Experiments

```bash
chatgot experiment --config-path configs/experiments/example.yaml
```

## Configuration Management

We use Hydra for configuration management. Key configuration files:

- `configs/models/`: Model architectures and hyperparameters
- `configs/data/`: Data processing and tokenization settings
- `configs/experiments/`: Experiment configurations and comparisons

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
isort .
```

## Experiment Tracking

We use MLflow for tracking:
- Training metrics
- Model parameters
- System resources
- Generated samples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.