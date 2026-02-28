"""Memory Calculator — Will This Model Fit?

Comprehensive memory estimator that covers inference and training,
multiple precisions, and all memory components (parameters, optimizer,
gradients, activations, KV cache).

No GPU required — pure math calculations.

Blog post: https://dadops.co/blog/gpu-memory-benchmarks/
Code Block 6 from the blog.
"""


def will_it_fit(
    param_billions,
    precision="fp16",
    training=True,
    optimizer="adam",
    batch_size=1,
    seq_length=2048,
    hidden_dim=4096,
    num_layers=32,
    gpu_vram_gb=24,
):
    """Estimate total GPU memory and whether it fits."""
    bpp = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}[precision]
    params = param_billions * 1e9

    # Model parameters
    param_mem = params * bpp

    # KV cache (inference only — during training, this is part of activations)
    num_heads = hidden_dim // 128  # Standard head_dim = 128
    kv_per_token = 2 * num_layers * num_heads * 128 * bpp
    kv_cache = kv_per_token * batch_size * seq_length if not training else 0

    # Optimizer states
    if training and optimizer == "adam":
        opt_mem = params * 4 * 3  # FP32 master + momentum + variance
    elif training and optimizer == "sgd":
        opt_mem = params * 4      # FP32 master weights only
    else:
        opt_mem = 0

    # Gradients
    grad_mem = params * bpp if training else 0

    # Activations (training only — rough estimate for transformers)
    if training:
        act_mem = 12 * hidden_dim * seq_length * num_layers * bpp * batch_size
    else:
        act_mem = 0

    total = param_mem + opt_mem + grad_mem + act_mem + kv_cache
    total_gb = total / 1e9
    fits = total_gb <= gpu_vram_gb

    components = {
        "Parameters": param_mem / 1e9,
        "Optimizer States": opt_mem / 1e9,
        "Gradients": grad_mem / 1e9,
        "Activations": act_mem / 1e9,
        "KV Cache": kv_cache / 1e9,
    }

    print(f"\n{'='*50}")
    print(f"  {param_billions}B model | {precision.upper()} | "
          f"{'Training' if training else 'Inference'}")
    print(f"  Batch={batch_size}, Seq={seq_length}")
    print(f"{'='*50}")
    for name, gb in components.items():
        if gb > 0:
            bar = "#" * int(gb * 2)
            print(f"  {name:<18} {gb:>7.1f} GB  {bar}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<18} {total_gb:>7.1f} GB")
    print(f"  GPU VRAM:          {gpu_vram_gb:>7.1f} GB")
    print(f"  Status:            {'✓ FITS' if fits else '✗ OOM'}")
    return total_gb, fits


if __name__ == "__main__":
    # Inference scenarios
    print("\n" + "=" * 60)
    print("INFERENCE SCENARIOS")
    print("=" * 60)
    will_it_fit(7, "fp16", training=False, batch_size=1, seq_length=512, gpu_vram_gb=24)
    will_it_fit(7, "fp16", training=False, batch_size=8, seq_length=2048, gpu_vram_gb=24)
    will_it_fit(7, "int4", training=False, batch_size=1, seq_length=2048, gpu_vram_gb=24)

    # Training scenarios
    print("\n" + "=" * 60)
    print("TRAINING SCENARIOS")
    print("=" * 60)
    will_it_fit(7, "fp16", training=True, optimizer="adam", batch_size=4, seq_length=512, gpu_vram_gb=80)
    will_it_fit(7, "fp16", training=True, optimizer="adam", batch_size=1, seq_length=2048, gpu_vram_gb=80)
    will_it_fit(1.3, "fp16", training=True, optimizer="adam", batch_size=8, seq_length=512, gpu_vram_gb=24)
