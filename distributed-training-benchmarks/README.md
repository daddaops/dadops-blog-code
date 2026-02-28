# Distributed Training Benchmarks

Verified, runnable code from the DadOps blog post:
**[Distributed Training Benchmarks: Data Parallel, Model Parallel, and Pipeline Parallel Compared](https://dadops.co/blog/distributed-training-benchmarks/)**

## Scripts

| Script | Description | GPU Required? |
|--------|-------------|---------------|
| `pipeline_simulation.py` | Simulates GPipe pipeline parallelism bubble ratios | No |
| `benchmark_table.py` | Prints comparison table of DDP/TP/PP/FSDP benchmarks | No |
| `decision_framework.py` | Recommends distributed training strategy based on model size | No |
| `fsdp_deepspeed_config.py` | Computes ZeRO memory reduction per stage | No |
| `ddp_benchmark.py` | Benchmarks DDP throughput on ResNet-50 | Yes (multi-GPU) |
| `tensor_parallel.py` | Column/row parallel linear layer implementations | Yes (multi-GPU) |

## Quick Start

```bash
# Pure Python scripts (no dependencies needed):
python3 pipeline_simulation.py
python3 benchmark_table.py
python3 decision_framework.py
python3 fsdp_deepspeed_config.py

# GPU scripts (requires CUDA + PyTorch):
torchrun --nproc_per_node=4 ddp_benchmark.py
```

## Output

Captured output from verified runs is in the `output/` directory.
