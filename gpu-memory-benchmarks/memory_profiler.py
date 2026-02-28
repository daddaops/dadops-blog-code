"""GPU Memory Profiler — Context Manager for Stage-by-Stage Profiling

A profiling harness using PyTorch's CUDA memory APIs to capture
before/after/peak memory at each stage of a model's lifecycle.

REQUIRES: NVIDIA GPU with CUDA support.

Blog post: https://dadops.co/blog/gpu-memory-benchmarks/
Code Block 1 from the blog.
"""

import torch
import contextlib


@contextlib.contextmanager
def gpu_memory_tracker(stage_name, log):
    """Track GPU memory before, during, and after a stage."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    yield  # Run the profiled code

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()

    log.append({
        "stage": stage_name,
        "before_mb": mem_before / 1e6,
        "after_mb": mem_after / 1e6,
        "peak_mb": mem_peak / 1e6,
        "delta_mb": (mem_after - mem_before) / 1e6,
    })


def profile_model(model_cls, input_fn, optimizer_cls=None):
    """Profile memory across model lifecycle stages."""
    log = []

    with gpu_memory_tracker("Model Loading", log):
        model = model_cls().cuda()

    x = input_fn()  # Create input on GPU

    with gpu_memory_tracker("Forward Pass", log):
        output = model(x)
        loss = output.sum()

    if optimizer_cls:
        optimizer = optimizer_cls(model.parameters())
        with gpu_memory_tracker("Backward Pass", log):
            loss.backward()

        with gpu_memory_tracker("Optimizer Step", log):
            optimizer.step()
            optimizer.zero_grad()

    # Print breakdown table
    print(f"{'Stage':-<20} {'Before':>10} {'After':>10} {'Peak':>10} {'Delta':>10}")
    print("-" * 62)
    for entry in log:
        print(f"{entry['stage']:-<20} {entry['before_mb']:>9.1f}M "
              f"{entry['after_mb']:>9.1f}M {entry['peak_mb']:>9.1f}M "
              f"{entry['delta_mb']:>+9.1f}M")
    return log


if __name__ == "__main__":
    print("This script requires an NVIDIA GPU with CUDA support.")
    print("Usage: import and call profile_model() with your model class and input function.")
