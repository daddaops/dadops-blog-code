"""Layer Normalization: normalize across the feature dimension.

Shows that the same token produces identical outputs regardless
of batch composition (batch-independent).
"""
import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer Normalization: normalize across the feature dimension."""
    # x shape: (batch_size, features) or (batch, seq, features)
    mean = x.mean(axis=-1, keepdims=True)     # Mean per token
    var  = x.var(axis=-1, keepdims=True)      # Variance per token
    x_hat = (x - mean) / np.sqrt(var + eps)   # Normalize
    return gamma * x_hat + beta               # Scale and shift

# Same test: same token with different batch companions
np.random.seed(42)
token = np.array([[1.0, 2.0, 3.0, 4.0]])

batch1 = np.vstack([token, np.random.randn(3, 4) * 0.1])
batch2 = np.vstack([token, np.random.randn(3, 4) * 10.0])

gamma4 = np.ones(4)
beta4  = np.zeros(4)

out1 = layer_norm(batch1, gamma4, beta4)[0]
out2 = layer_norm(batch2, gamma4, beta4)[0]

print(f"Same token:        {token[0]}")
print(f"Output (batch 1):  [{', '.join(f'{v:.4f}' for v in out1)}]")
print(f"Output (batch 2):  [{', '.join(f'{v:.4f}' for v in out2)}]")
print(f"Identical outputs? {np.allclose(out1, out2)}")
