"""Decision Framework — Choosing Your Distributed Training Strategy

A function that recommends a distributed training strategy based on
model size, GPU memory, number of GPUs, and interconnect type.

Blog post: https://dadops.co/blog/distributed-training-benchmarks/
Code Block 7 from the blog.
"""


def recommend_strategy(
    param_billions: float,
    gpu_mem_gb: float = 80.0,
    num_gpus: int = 8,
    interconnect: str = "nvlink",  # "nvlink", "infiniband", or "pcie"
) -> dict:
    """Recommend a distributed training strategy based on hardware constraints.

    Returns a dict with 'strategy', 'reason', and 'config_tips'.
    """
    # Static training memory: params(fp16) + grads(fp16) + adam(fp32)
    # = 2B + 2B + 8B = 12 bytes per parameter
    static_mem_gb = param_billions * 12

    # Activations typically need 3-5× the static memory for realistic
    # batch sizes, so we use conservative thresholds for GPU headroom.
    fits_one_gpu = static_mem_gb < gpu_mem_gb * 0.15
    fits_with_ckpt = static_mem_gb < gpu_mem_gb * 0.25

    # Per-layer memory: rough estimate — largest layer is ~4× param/num_layers
    # Assume 32 layers for simplicity
    largest_layer_gb = (param_billions * 2 * 4) / 32

    if fits_one_gpu:
        return {
            "strategy": "DDP",
            "reason": f"Static memory ~{static_mem_gb:.0f} GB fits in "
                      f"{gpu_mem_gb:.0f} GB with room for activations. "
                      f"DDP gives best throughput.",
            "config_tips": "torchrun --nproc_per_node=N, batch_size × N",
        }

    if fits_with_ckpt:
        return {
            "strategy": "DDP + Activation Checkpointing",
            "reason": f"Tight fit ({static_mem_gb:.0f} GB static). "
                      f"Checkpointing frees activation memory at "
                      f"~33% compute cost.",
            "config_tips": "torch.utils.checkpoint.checkpoint() on each block",
        }

    sharded_mem = static_mem_gb / num_gpus
    if largest_layer_gb < gpu_mem_gb * 0.5 and sharded_mem < gpu_mem_gb * 0.7:
        return {
            "strategy": "FSDP2 (ZeRO-3)",
            "reason": f"Static memory {static_mem_gb:.0f} GB total, "
                      f"~{sharded_mem:.1f} GB/GPU after sharding across "
                      f"{num_gpus} GPUs.",
            "config_tips": "fully_shard() each transformer block, then model",
        }

    if interconnect == "nvlink":
        return {
            "strategy": "TP (intra-node) + FSDP (across replicas)",
            "reason": f"Too large for FSDP alone ({sharded_mem:.0f} GB/GPU "
                      f"after sharding). TP splits layers across NVLink GPUs.",
            "config_tips": "DeviceMesh([tp_dim, dp_dim]), TP=4 + FSDP=2",
        }

    return {
        "strategy": "3D Parallelism (TP + PP + FSDP)",
        "reason": f"Large model ({param_billions}B) on {interconnect}. "
                  f"Full 3D parallelism needed.",
        "config_tips": "TP within node, PP across nodes, FSDP for replicas",
    }


if __name__ == "__main__":
    # Test it
    for size in [0.025, 0.125, 1.3, 7.0, 70.0]:
        r = recommend_strategy(size)
        print(f"\n{size}B params → {r['strategy']}")
        print(f"  Reason: {r['reason']}")
        print(f"  Config: {r['config_tips']}")
