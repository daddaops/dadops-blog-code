"""Grouped-Query Attention (GQA) — The Sweet Spot.

Divides query heads into groups, where each group shares one K,V pair.
Interpolates between MQA (1 group) and MHA (n_heads groups).
"""
import numpy as np
np.random.seed(42)

def grouped_query_attention(x, n_heads, d_head, n_groups):
    """GQA: heads are divided into groups sharing K, V."""
    seq_len, d_model = x.shape
    heads_per_group = n_heads // n_groups

    all_outputs = []
    kv_cache_size = 0

    for g in range(n_groups):
        # Each group has its own K, V
        W_K = np.random.randn(d_model, d_head) * 0.1
        W_V = np.random.randn(d_model, d_head) * 0.1
        K = x @ W_K
        V = x @ W_V
        kv_cache_size += K.size + V.size

        # Multiple query heads share this group's K, V
        for h in range(heads_per_group):
            W_Q = np.random.randn(d_model, d_head) * 0.1
            Q = x @ W_Q
            scores = Q @ K.T / np.sqrt(d_head)
            weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
            all_outputs.append(weights @ V)

    return np.concatenate(all_outputs, axis=-1), kv_cache_size

if __name__ == "__main__":
    # Compare all three: MHA, GQA, MQA
    seq_len, d_model, n_heads, d_head = 16, 64, 8, 8
    x = np.random.randn(seq_len, d_model)

    configs = [
        ("MHA  (g=8)", n_heads),   # every head has its own K,V
        ("GQA  (g=4)", 4),         # 4 groups, 2 heads per group
        ("GQA  (g=2)", 2),         # 2 groups, 4 heads per group
        ("MQA  (g=1)", 1),         # single shared K,V
    ]

    print(f"{'Variant':<14} {'KV cache':>10} {'Reduction':>10}")
    print("-" * 36)
    mha_cache = None
    for name, g in configs:
        _, cache = grouped_query_attention(x, n_heads, d_head, g)
        if mha_cache is None:
            mha_cache = cache
        ratio = mha_cache / cache
        print(f"{name:<14} {cache:>10} {ratio:>9.0f}x")
