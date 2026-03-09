import numpy as np

def grouped_query_attention_with_cache(x_new, K_cache, V_cache,
                                       W_q_heads, W_k_groups, W_v_groups,
                                       n_heads=32, n_kv_groups=8):
    """
    GQA: 32 query heads share 8 KV groups (4 query heads per group).
    Cache stores only 8 K and 8 V vectors per token, not 32.
    """
    head_dim = W_k_groups[0].shape[1]
    heads_per_group = n_heads // n_kv_groups  # 32 / 8 = 4

    # Project new token's K and V — only n_kv_groups projections, not n_heads
    k_groups = [x_new @ W_k_groups[g] for g in range(n_kv_groups)]
    v_groups = [x_new @ W_v_groups[g] for g in range(n_kv_groups)]

    # Append to cache (one K, V per group)
    new_K_cache = [np.vstack([K_cache[g], k_groups[g].reshape(1, -1)])
                   for g in range(n_kv_groups)]
    new_V_cache = [np.vstack([V_cache[g], v_groups[g].reshape(1, -1)])
                   for g in range(n_kv_groups)]

    # Each query head attends using its group's cached K and V
    head_outputs = []
    for h in range(n_heads):
        g = h // heads_per_group       # which KV group this head uses
        q_h = x_new @ W_q_heads[h]     # (1, head_dim)
        q_h = q_h.reshape(1, -1)

        scores = q_h @ new_K_cache[g].T / np.sqrt(head_dim)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = weights / weights.sum(axis=-1, keepdims=True)
        head_outputs.append((weights @ new_V_cache[g]).flatten())

    output = np.concatenate(head_outputs)
    return output, new_K_cache, new_V_cache

if __name__ == "__main__":
    np.random.seed(42)
    d_model = 128
    head_dim = 16
    n_heads = 32
    n_kv_groups = 8

    W_q_heads = [np.random.randn(d_model, head_dim) * 0.1 for _ in range(n_heads)]
    W_k_groups = [np.random.randn(d_model, head_dim) * 0.1 for _ in range(n_kv_groups)]
    W_v_groups = [np.random.randn(d_model, head_dim) * 0.1 for _ in range(n_kv_groups)]

    # Initialize cache with 5 tokens
    K_cache = [np.random.randn(5, head_dim) for _ in range(n_kv_groups)]
    V_cache = [np.random.randn(5, head_dim) for _ in range(n_kv_groups)]

    x_new = np.random.randn(d_model)
    out, K_cache, V_cache = grouped_query_attention_with_cache(
        x_new, K_cache, V_cache, W_q_heads, W_k_groups, W_v_groups)

    mha_cache = n_heads * head_dim * 2  # MHA: 32 KV heads
    gqa_cache = n_kv_groups * head_dim * 2  # GQA: 8 KV groups
    print(f"GQA output dim: {out.shape[0]}, cache entries per token: {gqa_cache} vs MHA: {mha_cache}")
    print(f"Cache reduction: {mha_cache / gqa_cache:.0f}x")
