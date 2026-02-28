"""FSDP2 vs DeepSpeed ZeRO — Configuration and Memory Comparison

Shows side-by-side configuration for PyTorch FSDP2 and DeepSpeed ZeRO,
plus memory reduction calculations for ZeRO stages on a 1.3B model.

REQUIRES: PyTorch with FSDP2 and/or DeepSpeed for actual execution.
The memory calculations below are pure math and run without dependencies.

Blog post: https://dadops.co/blog/distributed-training-benchmarks/
Code Block 6 from the blog.
"""


def compute_zero_memory(param_billions, num_gpus=8):
    """Compute memory per GPU for each ZeRO stage.

    Assumes fp16 params/grads, fp32 Adam optimizer states.
    """
    # Component sizes
    params_gb = param_billions * 2     # fp16: 2 bytes/param
    grads_gb = param_billions * 2      # fp16: 2 bytes/param
    adam_gb = param_billions * 8        # fp32 m + v: 4+4 bytes/param
    total_gb = params_gb + grads_gb + adam_gb

    results = {
        "DDP (baseline)": {
            "per_gpu_gb": total_gb,
            "reduction": "1.0×",
        },
        "ZeRO-1 (shard optimizer)": {
            "per_gpu_gb": params_gb + grads_gb + adam_gb / num_gpus,
            "reduction": f"{total_gb / (params_gb + grads_gb + adam_gb / num_gpus):.1f}×",
        },
        "ZeRO-2 (shard optimizer + grads)": {
            "per_gpu_gb": params_gb + (grads_gb + adam_gb) / num_gpus,
            "reduction": f"{total_gb / (params_gb + (grads_gb + adam_gb) / num_gpus):.1f}×",
        },
        "ZeRO-3 (shard all)": {
            "per_gpu_gb": total_gb / num_gpus,
            "reduction": f"{num_gpus:.1f}×",
        },
    }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Memory comparison: 1.3B-param model with Adam optimizer")
    print(f"{'':>38} {'Per GPU':>10} {'Reduction':>10}")
    print("=" * 60)

    results = compute_zero_memory(1.3, num_gpus=8)
    for stage, data in results.items():
        print(f"  {stage:<36} {data['per_gpu_gb']:>8.1f} GB  {data['reduction']:>8}")

    print()
    print("Note: activations add 5-20 GB depending on batch size and")
    print("sequence length. Use activation checkpointing to trade")
    print("compute for memory when needed.")

    # Verify against blog's claimed numbers:
    # DDP:    15.6 GB/GPU
    # ZeRO-1:  6.5 GB/GPU (2.4×)
    # ZeRO-2:  4.2 GB/GPU (3.7×)
    # ZeRO-3:  2.0 GB/GPU (7.8×)
