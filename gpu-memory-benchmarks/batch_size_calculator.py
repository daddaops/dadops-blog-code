"""Batch Size Calculator — Find Max Batch Size for GPU

Calculates the maximum batch size that fits in GPU memory by
decomposing memory into fixed (params + optimizer + gradients) and
variable (activations) components.

No GPU required — uses formulas with mock model parameters.

Blog post: https://dadops.co/blog/gpu-memory-benchmarks/
Code Block 4 from the blog.
"""


def find_max_batch_size(param_count, hidden_size, num_layers,
                        seq_len, gpu_memory_gb, precision="fp16"):
    """Find the maximum batch size that fits in GPU memory.

    Standalone version that takes model parameters directly instead of
    requiring a torch model object.
    """
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}[precision]

    # Fixed memory: params + gradients
    param_mem = param_count * bytes_per_param
    grad_mem = param_count * bytes_per_param

    # Optimizer states (Adam: FP32 master weights + 2 momentum buffers)
    opt_mem = param_count * 4 * 3  # 3 FP32 copies

    fixed_mem = param_mem + grad_mem + opt_mem
    available_for_activations = (gpu_memory_gb * 1e9) - fixed_mem

    # Activation memory per sample (rough estimate for transformers):
    # ~12 * hidden_dim * seq_len * num_layers * bytes_per_param
    act_per_sample = 12 * hidden_size * seq_len * num_layers * bytes_per_param

    max_batch = int(available_for_activations / act_per_sample)
    max_batch = max(max_batch, 0)

    print(f"Fixed memory:  {fixed_mem / 1e9:.1f} GB "
          f"(params={param_mem/1e9:.1f}, grads={grad_mem/1e9:.1f}, "
          f"optimizer={opt_mem/1e9:.1f})")
    print(f"Activation/sample: {act_per_sample / 1e6:.0f} MB")
    print(f"Available for activations: {available_for_activations / 1e9:.1f} GB")
    print(f"Max batch size: {max_batch}")

    return max_batch


if __name__ == "__main__":
    # Test with common model configurations
    configs = [
        ("7B (FP16, Adam, 80GB GPU)", 7e9, 4096, 32, 512, 80, "fp16"),
        ("7B (FP16, Adam, 24GB GPU)", 7e9, 4096, 32, 512, 24, "fp16"),
        ("13B (FP16, Adam, 80GB GPU)", 13e9, 5120, 40, 512, 80, "fp16"),
        ("1.3B (FP16, Adam, 24GB GPU)", 1.3e9, 2048, 24, 512, 24, "fp16"),
    ]

    for name, params, hidden, layers, seq, vram, prec in configs:
        print(f"\n--- {name} ---")
        find_max_batch_size(params, hidden, layers, seq, vram, prec)
