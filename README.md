# ChatGoT: A Text Generation Pipeline

ChatGoT is a comprehensive framework for training and using text generation models, featuring a complete pipeline from data preprocessing to text generation.

## Features

- **Complete Pipeline**: End-to-end text generation solution
- **Modular Architecture**: Easily customizable components
- **Performance Optimization**: Built-in throughput monitoring and optimization
- **Comprehensive Benchmarking**: Tools for measuring and comparing model performance
- **Visualization Tools**: Real-time performance visualization

## Architecture

The project is organized into several core modules:

- `src/`: Main source code for model, training, and data handling
- `pipeline/`: Orchestration system for running the complete pipeline
- `benchmarking/`: Performance measurement and comparison tools

## Installation

```bash
git clone https://github.com/yourusername/ChatGoT.git
cd ChatGoT
pip install -r requirements.txt
```

## Usage

### Running the Pipeline

The pipeline can be run from end-to-end or in specific stages:

```bash
python pipeline/main.py --input-file data/input.txt --pipeline-dir runs/my_run
```

Key options:
- `--stage [process|optimize|train|generate]`: Start from a specific stage
- `--resume`: Resume from last completed stage
- `--force-restart`: Start from the beginning, ignoring previous runs
- `--skip-process`: Skip the data processing stage
- `--skip-optimization`: Skip the hyperparameter optimization stage

### Running Benchmarks

The benchmarking system provides detailed performance metrics:

```bash
python benchmarking/runner.py --output-dir benchmark_results/my_benchmark
```

Key options:
- `--benchmarks [training_speed inference_performance model_accuracy]`: Specify which benchmarks to run
- `--baseline-file previous_benchmark.json`: Compare with previous results
- `--model-checkpoint runs/my_run/train/model.pt`: Model to benchmark

## Benchmarking

The benchmarking system measures and reports on:

### Training Performance
- Throughput (samples/second)
- Memory usage
- Component timing breakdown
- Optimal batch size determination

### Inference Performance
- Token generation speed
- First token and subsequent token latency
- Batch size scaling
- Memory usage during generation

### Model Accuracy
- Perplexity on validation data
- Token prediction accuracy
- Text quality metrics

## Pipeline Architecture

The pipeline consists of four main stages:

1. **Process**: Data preprocessing, tokenization, and dataset creation
2. **Optimize**: Hyperparameter optimization for best model configuration
3. **Train**: Model training with optimized parameters
4. **Generate**: Text generation with the trained model

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Organization](docs/organization.md): Overview of the project structure and components
- [Optimizations](docs/optimizations.md): Performance optimization techniques implemented in the project

## Monitoring

Real-time monitoring is available during training:

- Performance metrics (samples/second)
- Memory usage tracking
- Component-level timing (data loading, forward pass, backward pass, etc.)
- Interactive visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.