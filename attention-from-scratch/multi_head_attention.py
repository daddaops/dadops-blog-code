"""Multi-Head Attention from Scratch.

Implements multi-head attention with the reshape trick: project once into
full d_model space, reshape to split into heads, run attention in parallel.
"""
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """Single-head attention."""
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = e / np.sum(e, axis=-1, keepdims=True)
    output = weights @ V
    return output, weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model     # e.g., 64
        self.num_heads = num_heads  # e.g., 8
        self.d_k = d_model // num_heads  # e.g., 8

        # Projection matrices (in a real model, these are learned)
        scale = 0.3
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, X):
        """
        X: (n, d_model) — input embeddings
        Returns: (n, d_model) — attended output
        """
        n = X.shape[0]

        # Project into Q, K, V — still full d_model width
        Q = X @ self.W_q   # (n, d_model)
        K = X @ self.W_k   # (n, d_model)
        V = X @ self.W_v   # (n, d_model)

        # Reshape to split heads: (n, d_model) -> (n, h, d_k) -> (h, n, d_k)
        Q = Q.reshape(n, self.num_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(n, self.num_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(n, self.num_heads, self.d_k).transpose(1, 0, 2)
        # Now Q, K, V are each (h, n, d_k) — h independent attention problems

        # Scaled dot-product attention for all heads at once
        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(self.d_k)  # (h, n, n)

        # Softmax per head (stable)
        e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = e / np.sum(e, axis=-1, keepdims=True)  # (h, n, n)

        # Weighted values
        attended = weights @ V           # (h, n, d_k)

        # Concatenate heads: (h, n, d_k) -> (n, h, d_k) -> (n, d_model)
        attended = attended.transpose(1, 0, 2).reshape(n, self.d_model)

        # Final linear projection
        output = attended @ self.W_o     # (n, d_model)

        return output, weights

if __name__ == "__main__":
    np.random.seed(42)

    n, d_model, num_heads = 6, 64, 8
    X = np.random.randn(n, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output, all_weights = mha.forward(X)

    print(f"Input shape:  {X.shape}")           # (6, 64)
    print(f"Output shape: {output.shape}")       # (6, 64)
    print(f"Weight shape: {all_weights.shape}")  # (8, 6, 6) — 8 attention matrices

    # Each head learns different patterns
    for h in range(min(3, num_heads)):
        print(f"\nHead {h} — where does token 0 attend?")
        w = all_weights[h, 0]
        for i, weight in enumerate(w):
            bar = "█" * int(weight * 40)
            print(f"  token {i}: {weight:.3f} {bar}")
