"""Scaled Dot-Product Attention from Scratch.

Implements the core attention formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
Then runs it on a toy sentence with random projections.
"""
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: arrays of shape (n, d_k)
    Returns: output (n, d_k), attention weights (n, n)
    """
    d_k = Q.shape[-1]

    # Step 1-2: Compute scaled scores
    scores = (Q @ K.T) / np.sqrt(d_k)       # (n, n)

    # Step 3: Softmax over keys (last axis)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = e / np.sum(e, axis=-1, keepdims=True)  # (n, n)

    # Step 4: Weighted sum of values
    output = weights @ V                      # (n, d_k)

    return output, weights

if __name__ == "__main__":
    np.random.seed(42)

    # Five tokens with 8-dimensional embeddings
    tokens = ["The", "cat", "sat", "on", "mat"]
    n, d_model, d_k = 5, 8, 8

    # Random token embeddings (in a real model, these are learned)
    embeddings = np.random.randn(n, d_model)

    # Random projection matrices (in a real model, these are learned too)
    W_q = np.random.randn(d_model, d_k) * 0.3
    W_k = np.random.randn(d_model, d_k) * 0.3
    W_v = np.random.randn(d_model, d_k) * 0.3

    # Project embeddings into Q, K, V spaces
    Q = embeddings @ W_q   # (5, 8)
    K = embeddings @ W_k   # (5, 8)
    V = embeddings @ W_v   # (5, 8)

    # Run attention
    output, weights = scaled_dot_product_attention(Q, K, V)

    # Print the attention weight matrix
    print("Attention weights (rows=queries, cols=keys):\n")
    print("       ", "  ".join(f"{t:>5s}" for t in tokens))
    for i, row in enumerate(weights):
        print(f"{tokens[i]:>5s}:", "  ".join(f"{w:.3f}" for w in row))
