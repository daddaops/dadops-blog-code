"""Batch Normalization: normalize across the batch dimension.

Shows that batch norm zeros the mean and standardizes to unit variance,
then demonstrates the batch dependence problem.
"""
import numpy as np


def batch_norm(x, gamma, beta, eps=1e-5):
    """Batch Normalization: normalize across the batch dimension."""
    # x shape: (batch_size, features)
    mean = x.mean(axis=0, keepdims=True)     # Mean per feature
    var  = x.var(axis=0, keepdims=True)      # Variance per feature
    x_hat = (x - mean) / np.sqrt(var + eps)  # Normalize
    return gamma * x_hat + beta              # Scale and shift

# Test it
np.random.seed(42)
x = np.random.randn(32, 128) * 5 + 3   # Shifted, scaled input

gamma = np.ones(128)     # Learnable scale (init: 1)
beta  = np.zeros(128)    # Learnable shift (init: 0)

out = batch_norm(x, gamma, beta)
print(f"Before BatchNorm:  mean = {x.mean():.4f}, std = {x.std():.4f}")
print(f"After  BatchNorm:  mean = {out.mean():.4f}, std = {out.std():.4f}")

# --- Batch dependence problem ---
print()
np.random.seed(42)
token = np.array([[1.0, 2.0, 3.0, 4.0]])

# Batch 1: token alongside small-valued companions
batch1 = np.vstack([token, np.random.randn(3, 4) * 0.1])

# Batch 2: same token alongside large-valued companions
batch2 = np.vstack([token, np.random.randn(3, 4) * 10.0])

gamma4 = np.ones(4)
beta4  = np.zeros(4)

out1 = batch_norm(batch1, gamma4, beta4)[0]  # First row = our token
out2 = batch_norm(batch2, gamma4, beta4)[0]

print(f"Same token:        {token[0]}")
print(f"Output (batch 1):  [{', '.join(f'{v:.4f}' for v in out1)}]")
print(f"Output (batch 2):  [{', '.join(f'{v:.4f}' for v in out2)}]")
print(f"Different outputs? {not np.allclose(out1, out2)}")
