"""
VRAM estimation function from the blog post.

From: https://dadops.dev/blog/local-llm-deployment/

Calculates memory requirements for local LLM inference:
  VRAM ≈ Model Weights + KV Cache + Overhead

No external dependencies required.
"""


def estimate_vram_gb(
    params_billions: float,
    bits_per_weight: float,
    context_length: int,
    overhead_gb: float = 0.8
) -> dict:
    """Estimate VRAM needed for local LLM inference."""
    # Model weights: params * bits / 8, converted to GB (decimal)
    weight_gb = (params_billions * 1e9 * bits_per_weight) / (8 * 1e9)
    # Add ~25% for embedding tables, layer norms, and GGUF metadata
    weight_gb *= 1.25

    # KV cache heuristic: ~0.5 GB per 7B params per 4K context
    kv_gb = (params_billions / 7) * (context_length / 4096) * 0.5

    total = weight_gb + kv_gb + overhead_gb
    return {
        "weights_gb": round(weight_gb, 1),
        "kv_cache_gb": round(kv_gb, 1),
        "overhead_gb": overhead_gb,
        "total_gb": round(total, 1)
    }


if __name__ == "__main__":
    # Examples from the blog post
    configs = [
        ("3B Q4",    3,  4.0,  4096),
        ("7B Q4",    7,  4.0,  4096),
        ("7B Q8",    7,  8.0,  4096),
        ("13B Q4",  13,  4.0,  4096),
        ("7B Q4 32K", 7, 4.0, 32768),
        ("70B Q4",  70,  4.0,  4096),
    ]

    print(f"{'Config':<14} {'Weights':>8} {'KV Cache':>9} {'Total':>7}")
    print("-" * 42)
    for name, params, bits, ctx in configs:
        r = estimate_vram_gb(params, bits, ctx)
        print(f"{name:<14} {r['weights_gb']:>7.1f}G {r['kv_cache_gb']:>8.1f}G {r['total_gb']:>6.1f}G")

    # Blog table claims to verify:
    # 3B Q4:     Weights=1.9, KV=0.2, Total=2.9
    # 7B Q4:     Weights=4.4, KV=0.5, Total=5.7
    # 7B Q8:     Weights=8.8, KV=0.5, Total=10.1
    # 13B Q4:    Weights=8.1, KV=0.9, Total=9.8
    # 7B Q4 32K: Weights=4.4, KV=4.0, Total=9.2
    # 70B Q4:    Weights=43.8, KV=5.0, Total=49.6
