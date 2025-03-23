# ChatGoT - Character-level GPT for Game of Thrones Text Generation

ChatGoT is a transformer-based text generation model trained on Game of Thrones content, implementing a character-level language model with a modern training pipeline.

## Features

- Character-level language modeling for more precise text generation
- Memory-efficient transformer architecture with optimized attention mechanisms
- Comprehensive training pipeline with automatic optimization
- GPU acceleration with mixed precision support
- Flexible text generation with temperature control

## Project Structure

```
ChatGoT/
├── pipeline.py                # Main entry point for all operations
├── run_tests.py               # Test runner script
├── requirements.txt           # Project dependencies
├── src/                       # Source code modules
│   ├── data_handler.py        # Data loading and processing
│   ├── data_processor.py      # Raw text processing
│   ├── model.py               # Transformer model architecture
│   ├── trainer.py             # Model training logic
│   ├── utils.py               # Utility functions
│   ├── batch_size_finder.py   # Batch size optimization
│   ├── cuda_optimizations.py  # CUDA-specific optimizations
│   └── text_generator.py      # Text generation utilities
├── tests/                     # Testing framework
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── data/                      # Raw data directory
├── processed_data/            # Processed data storage
└── backup/                    # Deprecated scripts (for reference)
```

## Installation

1. Ensure you have Python 3.8+ and PyTorch 2.0+ installed
2. Clone this repository
3. Install dependencies: `pip install -r requirements.txt`

## Usage

### Process Raw Data

```bash
python pipeline.py --stage process --input_file data/your_text_file.txt --processed_data_path processed_data/output.pkl
```

### Optimize Training Parameters

```bash
python pipeline.py --stage optimize --processed_data_path processed_data/your_data.pkl
```

### Train the Model

```bash
python pipeline.py --stage train --processed_data_path processed_data/your_data.pkl
```

### Generate Text

```bash
python pipeline.py --stage generate --processed_data_path processed_data/your_data.pkl --generate_seed "In the seven kingdoms, " --generate_length 200
```

### Run the Full Pipeline

```bash
python pipeline.py --input_file data/your_text_file.txt
```

### Skip Stages

You can skip stages if you've already completed them:

```bash
python pipeline.py --skip_process --processed_data_path processed_data/your_data.pkl
```

## Testing

Run all tests:

```bash
python run_tests.py
```

## Model Architecture

ChatGoT uses a standard GPT-2 Small architecture with:
- Character-level embeddings
- Multi-head self-attention
- Transformer encoder layers
- Memory-efficient implementation options

## Performance Optimization

The pipeline automatically:
- Determines optimal batch size
- Enables mixed precision when available
- Optimizes CUDA operations
- Implements gradient accumulation for larger effective batch sizes

For advanced optimization techniques and experiments, see [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

## Requirements

See `requirements.txt` for full dependencies.

## License

[MIT License](LICENSE)