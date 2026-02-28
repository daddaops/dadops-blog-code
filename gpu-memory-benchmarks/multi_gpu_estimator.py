"""Multi-GPU Memory Distribution — Strategy Comparison

Estimates per-GPU memory for different parallelism strategies:
data parallel, pipeline parallel, and FSDP/ZeRO-3.

No GPU required — pure math calculations.

Blog post: https://dadops.co/blog/gpu-memory-benchmarks/
Code Block 5 from the blog.
"""


def estimate_multi_gpu_memory(param_count_b, num_gpus, strategy, precision="fp16"):
    """Estimate per-GPU memory for different parallelism strategies."""
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}[precision]
    param_mem = param_count_b * 1e9 * bytes_per_param
    grad_mem = param_mem  # Same as params
    opt_mem = param_count_b * 1e9 * 4 * 3  # Adam: FP32 master + m + v

    if strategy == "data_parallel":
        # Each GPU holds a FULL copy of everything, batches are split
        per_gpu_mem = param_mem + grad_mem + opt_mem
        comm = "AllReduce gradients every step"
        speedup = f"~{num_gpus}x throughput (linear scaling)"

    elif strategy == "pipeline_parallel":
        # Model layers split across GPUs
        per_gpu_mem = (param_mem + grad_mem + opt_mem) / num_gpus
        comm = "Activations sent between pipeline stages"
        speedup = "~{0}x capacity, <{0}x throughput (pipeline bubbles)".format(
            num_gpus)

    elif strategy == "fsdp_zero3":
        # Params, gradients, AND optimizer sharded across GPUs
        per_gpu_mem = (param_mem + grad_mem + opt_mem) / num_gpus
        comm = "AllGather params before forward, ReduceScatter gradients"
        speedup = f"~{num_gpus}x capacity, good throughput with overlap"

    per_gpu_gb = per_gpu_mem / 1e9
    print(f"Strategy: {strategy}")
    print(f"  Per-GPU memory: {per_gpu_gb:.1f} GB (model/grads/optimizer)")
    print(f"  Communication: {comm}")
    print(f"  Scaling: {speedup}")
    return per_gpu_gb


if __name__ == "__main__":
    print("=" * 60)
    print("Multi-GPU Memory Comparison: 13B model (FP16 + Adam)")
    print("=" * 60)

    for strategy in ["data_parallel", "pipeline_parallel", "fsdp_zero3"]:
        for gpus in [2, 4]:
            print(f"\n--- {strategy}, {gpus} GPUs ---")
            estimate_multi_gpu_memory(13, gpus, strategy)
