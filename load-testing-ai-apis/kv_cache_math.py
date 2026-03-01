"""
KV Cache Memory Calculation — verifies the blog's memory math.

From: https://dadops.dev/blog/load-testing-ai-apis/
Code Block 1: "Why AI APIs Break Differently"

Blog claims (corrected):
  - Llama-3 70B (FP16): 327,680 bytes per token for KV cache
  - 8K context: 2.50 GB per request
  - 32 concurrent requests: 80 GB (fills an 80 GB A100)
"""


def kv_cache_bytes_per_token(num_layers, num_kv_heads, head_dim,
                              bytes_per_param):
    """Calculate KV cache memory per token per sequence.

    Formula: 2 * num_layers * num_kv_heads * head_dim * bytes_per_param
    The factor of 2 accounts for both K and V caches.
    """
    return 2 * num_layers * num_kv_heads * head_dim * bytes_per_param


if __name__ == "__main__":
    # Llama-3 70B parameters (FP16)
    num_layers = 80
    num_kv_heads = 8       # GQA: 8 KV heads (vs 64 attention heads)
    head_dim = 128
    bytes_per_param = 2    # FP16 = 2 bytes

    kv_bytes = kv_cache_bytes_per_token(num_layers, num_kv_heads,
                                         head_dim, bytes_per_param)

    print("=== KV Cache Memory Verification (Llama-3 70B, FP16) ===")
    print(f"  Formula: 2 * {num_layers} * {num_kv_heads} * {head_dim} * {bytes_per_param}")
    print(f"  Per token: {kv_bytes:,} bytes")
    print(f"  Blog claims: 327,680 bytes")
    print(f"  Match: {kv_bytes == 327_680}")

    # 8K context
    context_len = 8192
    per_request_bytes = kv_bytes * context_len
    per_request_gb = per_request_bytes / (1024 ** 3)
    print(f"\n  8K context ({context_len} tokens):")
    print(f"    {per_request_bytes:,} bytes = {per_request_gb:.2f} GB")
    print(f"    Blog claims: 2.56 GB")
    # Note: 327680 * 8192 = 2,684,354,560 bytes = 2.50 GB (not 2.56 GB)
    # Blog may use 1 GB = 10^9 bytes: 2,684,354,560 / 10^9 = 2.68 GB
    # Or blog may round differently
    exact_gb_binary = per_request_bytes / (1024 ** 3)
    exact_gb_decimal = per_request_bytes / 1e9
    print(f"    Exact (binary GiB): {exact_gb_binary:.2f}")
    print(f"    Exact (decimal GB): {exact_gb_decimal:.2f}")

    # 32 concurrent requests
    concurrent = 32
    total_bytes = per_request_bytes * concurrent
    total_gb_binary = total_bytes / (1024 ** 3)
    total_gb_decimal = total_bytes / 1e9
    print(f"\n  {concurrent} concurrent requests:")
    print(f"    {total_bytes:,} bytes")
    print(f"    Binary GiB: {total_gb_binary:.1f}")
    print(f"    Decimal GB: {total_gb_decimal:.1f}")
    print(f"    Blog claims: 82 GB")
    print(f"    Exceeds 80 GB A100: {total_gb_decimal > 80}")
