"""Multi-Head Attention vs Multi-Query Attention.

Compares standard MHA (each head has its own K,V) with MQA (all heads
share a single K,V), showing the KV cache reduction.
"""
import numpy as np
np.random.seed(42)

def multi_head_attention(x, n_heads, d_head):
    """Standard MHA: each head has its own Q, K, V."""
    seq_len, d_model = x.shape

    all_outputs = []
    kv_cache_size = 0
    for h in range(n_heads):
        W_Q = np.random.randn(d_model, d_head) * 0.1
        W_K = np.random.randn(d_model, d_head) * 0.1
        W_V = np.random.randn(d_model, d_head) * 0.1

        Q = x @ W_Q    # (seq_len, d_head)
        K = x @ W_K
        V = x @ W_V
        kv_cache_size += K.size + V.size  # each head stores K, V

        scores = Q @ K.T / np.sqrt(d_head)
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        all_outputs.append(weights @ V)

    return np.concatenate(all_outputs, axis=-1), kv_cache_size

def multi_query_attention(x, n_heads, d_head):
    """MQA: all heads share a single K, V."""
    seq_len, d_model = x.shape

    # Single shared K, V
    W_K = np.random.randn(d_model, d_head) * 0.1
    W_V = np.random.randn(d_model, d_head) * 0.1
    K = x @ W_K
    V = x @ W_V
    kv_cache_size = K.size + V.size  # only one K, V pair

    all_outputs = []
    for h in range(n_heads):
        W_Q = np.random.randn(d_model, d_head) * 0.1
        Q = x @ W_Q
        scores = Q @ K.T / np.sqrt(d_head)
        weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        all_outputs.append(weights @ V)

    return np.concatenate(all_outputs, axis=-1), kv_cache_size

if __name__ == "__main__":
    # Compare on a toy sequence
    seq_len, d_model, n_heads, d_head = 16, 64, 8, 8
    x = np.random.randn(seq_len, d_model)

    mha_out, mha_cache = multi_head_attention(x, n_heads, d_head)
    mqa_out, mqa_cache = multi_query_attention(x, n_heads, d_head)

    print(f"MHA output shape: {mha_out.shape}, KV cache: {mha_cache} values")
    print(f"MQA output shape: {mqa_out.shape}, KV cache: {mqa_cache} values")
    print(f"Cache reduction: {mha_cache / mqa_cache:.0f}x")
