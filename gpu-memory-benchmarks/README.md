# GPU Memory Benchmarks

Verified, runnable code from the DadOps blog post:
**[GPU Memory Benchmarks: Will This Model Fit?](https://dadops.co/blog/gpu-memory-benchmarks/)**

## Scripts

| Script | Description | GPU Required? |
|--------|-------------|---------------|
| `batch_size_calculator.py` | Calculates max batch size given GPU VRAM | No |
| `multi_gpu_estimator.py` | Compares multi-GPU parallelism strategies | No |
| `will_it_fit.py` | Comprehensive memory calculator for any config | No |
| `memory_profiler.py` | Context manager for stage-by-stage GPU profiling | Yes |
| `inference_benchmark.py` | Measures inference memory across precisions | Yes (+ HuggingFace) |
| `training_profiler.py` | Profiles training memory waterfall | Yes |

## Quick Start

```bash
# Pure Python scripts (no dependencies needed):
python3 batch_size_calculator.py
python3 multi_gpu_estimator.py
python3 will_it_fit.py
```

## Output

Captured output from verified runs is in the `output/` directory.
