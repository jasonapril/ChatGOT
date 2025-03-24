# ChatGoT Project Organization

This document outlines the organization of the ChatGoT project, providing guidance on where various components are located and how they interact.

## Directory Structure

The project follows a modular organization with clearly separated concerns:

```
ChatGoT/
├── benchmarking/             # Benchmarking framework and results
│   ├── benchmarks/           # Individual benchmark definitions
│   ├── logs/                 # Benchmark runtime logs
│   ├── results/              # Benchmark result data and summaries
│   └── utils/                # Benchmark utilities and helpers
├── data/                     # Raw training data
├── pipeline/                 # Pipeline framework
│   ├── core/                 # Core pipeline functionality
│   ├── process/              # Data processing stage
│   ├── optimize/             # Optimization stage
│   ├── train/                # Training stage
│   ├── generate/             # Text generation stage
│   └── stages/               # Stage definitions
├── processed_data/           # Processed and prepared datasets
├── src/                      # Core library code
│   ├── monitoring/           # Performance monitoring
│   │   ├── throughput_core.py    # Core throughput monitoring
│   │   ├── instrumentation.py    # Model instrumentation
│   │   └── visualization.py      # Visualization tools
│   ├── training/             # Training components
│   │   ├── training_loop.py      # Core training loop
│   │   ├── evaluation.py         # Model evaluation
│   │   ├── generation.py         # Text generation
│   │   ├── optimizations.py      # Training optimizations
│   │   └── train_config.py       # Training configuration
│   └── benchmarks/           # Framework benchmark implementations
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── run_all_tests.py      # Unified test runner
└── requirements.txt          # Project dependencies
```

## Key Components

### Core Modules

1. **src/monitoring/**
   - Contains tools for performance monitoring, instrumentation, and visualization
   - `throughput_core.py`: Core classes for tracking training throughput
   - `instrumentation.py`: Instruments PyTorch models for detailed metrics
   - `visualization.py`: Creates visualizations of performance data

2. **src/training/**
   - Contains the model training infrastructure
   - `training_loop.py`: Efficient training loop implementation
   - `evaluation.py`: Model evaluation utilities
   - `generation.py`: Text generation functionality
   - `optimizations.py`: Performance optimizations for training
   - `train_config.py`: Training configuration management

### Pipeline Framework

The pipeline framework provides a modular, resumable training pipeline:

1. **pipeline/core/** - Central pipeline management
2. **pipeline/stages/** - Pipeline stage definitions
3. **pipeline/process**, **pipeline/optimize**, etc. - Stage-specific implementations

### Benchmarking

The benchmarking system provides tools for measuring and comparing performance:

1. **benchmarking/benchmarks/** - Specific benchmark definitions
2. **benchmarking/results/** - Benchmark result data and summaries
3. **benchmarking/logs/** - Detailed logs from benchmark runs

### Testing

The testing framework provides comprehensive test coverage:

1. **tests/unit/** - Unit tests for individual components
2. **tests/integration/** - Tests for component interactions
3. **tests/run_all_tests.py** - Unified test runner

## Usage Guidance

### Running Tests

Use the unified test runner:

```bash
python tests/run_all_tests.py
```

Run specific tests or test modules:

```bash
python tests/run_all_tests.py --test test_save_monitor_stats
python tests/run_all_tests.py --module test_instrumentation
```

### Running the Pipeline

Use the main pipeline interface:

```bash
python pipeline/main.py --input_file data/your_dataset.txt
```

Resume an interrupted pipeline:

```bash
python pipeline/main.py --resume
```

### Running Benchmarks

Use the benchmark runner:

```bash
python benchmarking/main.py
```

Run specific benchmarks:

```bash
python benchmarking/main.py --benchmark training_throughput
```

## Design Principles

The project follows these key design principles:

1. **Modularity**: Components are separated by concern and can be used independently
2. **Encapsulation**: Implementation details are hidden behind clean interfaces
3. **Testability**: All components are designed to be easily testable
4. **Instrumentation**: Performance metrics are collected throughout the system
5. **Configurability**: Components are configurable via clear parameter interfaces 