import numpy as np

def self_attention(X, W_q, W_k, W_v):
    """Minimal self-attention: X is (n_tokens, d_model)."""
    Q = X @ W_q   # queries
    K = X @ W_k   # keys
    V = X @ W_v   # values
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    return weights @ V

np.random.seed(42)
d = 4
W_q = np.random.randn(d, d) * 0.5
W_k = np.random.randn(d, d) * 0.5
W_v = np.random.randn(d, d) * 0.5

# 3 tokens, each of dimension 4
X = np.random.randn(3, d)

# Permutation: swap token 0 and token 2
P = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=float)

# Test: attention(P @ X) == P @ attention(X)?
out_original = self_attention(X, W_q, W_k, W_v)
out_permuted_input = self_attention(P @ X, W_q, W_k, W_v)
out_permuted_output = P @ out_original

print(f"Equivariant? {np.allclose(out_permuted_input, out_permuted_output)}")
# True! Self-attention commutes with permutations.
