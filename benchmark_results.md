# Benchmark Results: Standard AdamW vs 8-bit AdamW

This document summarizes the benchmarking results comparing the standard PyTorch AdamW optimizer with the 8-bit AdamW optimizer from bitsandbytes at various batch sizes.

## Benchmark Configuration
- Model: GPT-like Transformer 
- Dataset: Game of Thrones character-level dataset
- PyTorch version: 2.6.0+cu118
- CUDA: Available
- Test duration: 50 steps after 10 warm-up steps

## Results Table

| Batch Size | Standard AdamW (tokens/sec) | 8-bit AdamW (tokens/sec) | Difference (%) |
|------------|------------------------------|---------------------------|----------------|
| 16         | 32,324.16                   | 29,870.22                 | -7.59%         |
| 24         | 32,997.69                   | 31,973.56                 | -3.10%         |
| 32         | 32,978.47                   | 32,260.48                 | -2.18%         |
| 48         | 33,866.15                   | 33,592.35                 | -0.81%         |
| 64         | 33,982.72                   | 33,738.47                 | -0.72%         |
| 96         | 34,004.93                   | 33,810.79                 | -0.57%         |
| 128        | 19,677.57                   | 19,575.64                 | -0.52%         |

## Analysis

1. **Performance Gap**: The standard AdamW optimizer consistently outperformed the 8-bit AdamW optimizer across all batch sizes, but the performance gap narrowed as batch sizes increased.

2. **Memory Efficiency**: While we expected the 8-bit optimizer to handle larger batch sizes more efficiently, both optimizers showed similar performance characteristics even at large batch sizes (96, 128).

3. **Performance Drop at Batch Size 128**: Both optimizers experienced a significant performance drop (~42% reduction) when increasing from batch size 96 to 128, likely indicating we hit hardware memory or computational limits.

4. **Convergence**: Both optimizers showed similar training loss trends, suggesting that the 8-bit optimizer maintains optimization quality despite the quantization.

## Conclusions

1. For this specific model and dataset, the standard AdamW optimizer delivers slightly better throughput compared to the 8-bit AdamW optimizer.

2. The primary advantage of the 8-bit optimizer appears to be memory efficiency rather than throughput. In scenarios where model size is much larger or memory constraints are tighter, the 8-bit optimizer might show more significant advantages.

3. The diminishing performance gap at larger batch sizes suggests that the 8-bit optimizer becomes more competitive as batch sizes increase, potentially due to better memory utilization.

4. For production deployments where memory efficiency is critical, the minimal performance loss (~0.5-1% at large batch sizes) may be an acceptable trade-off for the reduced memory footprint.

## Next Steps

1. Measure actual GPU memory usage for both optimizers to quantify memory savings.

2. Test with larger models where memory constraints become more significant.

3. Evaluate training convergence over longer periods to ensure the 8-bit optimizer doesn't negatively impact final model quality.

4. Consider hybrid approaches, such as using the standard optimizer for critical layers and 8-bit for others. 