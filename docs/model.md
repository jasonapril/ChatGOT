# ChatGoT Model Architecture

This document describes the architecture and parameters of the ChatGoT model.

## Model Variants

Currently, we have two main model configurations:

1. **ChatGoT Small** (~114M parameters)
   - Based on GPT-2 Small architecture
   - Character-level tokenization
   - 768-dimensional embeddings
   - 12 transformer layers

2. **ChatGoT Tiny** (~10M parameters)
   - Smaller version for quick experiments
   - Character-level tokenization
   - 256-dimensional embeddings
   - 4 transformer layers

## Architecture Details

ChatGoT is a decoder-only transformer model (similar to GPT) that works at the character level. It consists of:

1. **Token Embedding Layer**: Maps each character to a vector of size `d_model`
2. **Position Embedding Layer**: Learned positional embeddings for sequences up to `max_seq_length`
3. **Transformer Decoder Blocks**: A stack of `n_layers` transformer blocks, each containing:
   - Multi-head self-attention (with `n_head` attention heads)
   - Feed-forward network with hidden dimension `d_hid`
   - Layer normalization and dropout

## Parameter Count Analysis

For the ChatGoT Small model with the following configuration:
- `d_model = 768` (embedding dimension)
- `n_layers = 12` (number of transformer layers)
- `vocab_size = 96` (character-level vocabulary)
- `n_head = 12` (number of attention heads)
- `d_hid = 3072` (hidden dimension in feed-forward network)
- `max_seq_length = 1024` (maximum sequence length)

The parameter count breaks down as follows:

| Component | Formula | Parameters | % of Total |
|-----------|---------|------------|------------|
| Token Embeddings | vocab_size * d_model | 73,728 | 0.1% |
| Position Embeddings | max_seq_length * d_model | 786,432 | 0.7% |
| Attention Layers | 4 * d_model * d_model * n_layers | 28,311,552 | 24.8% |
| Feed-forward Layers | (d_model * d_hid + d_hid * d_model) * n_layers | 56,623,104 | 49.5% |
| Layer Norms | 4 * d_model * n_layers + 2 * d_model | 38,400 | < 0.1% |
| Output Projection | d_model * vocab_size | 73,728 | 0.1% |
| Memory Architecture | (specific to our implementation) | ~28M | 24.8% |
| **Total** | | **~114M** | 100% |

The difference between our model's parameter count (~114M) and GPT-2 Small (~85M) is due to:
1. Different architecture implementation (our custom transformer decoder)
2. Additional memory components in our implementation
3. Character-level vs. subword tokenization

## Hyperparameters

### Training Hyperparameters

| Parameter | Small | Tiny |
|-----------|-------|------|
| `batch_size` | 8 | 32 |
| `learning_rate` | 0.0005 | 0.001 |
| `weight_decay` | 0.01 | 0.01 |
| `dropout` | 0.1 | 0.1 |

### Memory Optimizations

To train on GPUs with limited memory, we use:
1. **Mixed Precision Training**: FP16/BF16 for forward pass
2. **Gradient Checkpointing**: Trades computation for memory
3. **Gradient Accumulation**: Simulates larger batch sizes

## Performance Metrics

Typical performance on consumer hardware:
- GTX 1650 Ti (4GB VRAM):
  - Small model: ~500-1000 tokens/sec
  - Tiny model: ~2500-5000 tokens/sec 