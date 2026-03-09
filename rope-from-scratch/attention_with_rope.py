"""Scaled dot-product attention with RoPE."""
import numpy as np
from helpers import rope_frequencies, apply_rope, softmax


def attention_with_rope(X, W_q, W_k, W_v, freqs):
    """Scaled dot-product attention with RoPE.

    X: (seq_len, d_model) — input embeddings (no positional encoding added!)
    """
    seq_len = X.shape[0]
    Q = X @ W_q                      # (seq_len, d_head)
    K = X @ W_k
    V = X @ W_v

    # Apply RoPE to Q and K (not V!)
    for pos in range(seq_len):
        Q[pos] = apply_rope(Q[pos], pos, freqs)
        K[pos] = apply_rope(K[pos], pos, freqs)

    # Standard scaled dot-product attention
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)
    weights = softmax(scores)         # softmax along last axis
    return weights @ V                # (seq_len, d_head)


# Smoke test
np.random.seed(42)
seq_len, d_model, d_head = 8, 64, 64
X = np.random.randn(seq_len, d_model)
W_q = np.random.randn(d_model, d_head) * 0.1
W_k = np.random.randn(d_model, d_head) * 0.1
W_v = np.random.randn(d_model, d_head) * 0.1
freqs = rope_frequencies(d_head)

output = attention_with_rope(X, W_q, W_k, W_v, freqs)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {output.shape}")
print("Attention with RoPE completed successfully")
