"""Gradient flow comparison: plain vs residual blocks."""
import numpy as np

def gradient_through_plain_block(x, W1, W2):
    """Forward + backward through a plain block: y = relu(W2 @ relu(W1 @ x))"""
    z1 = W1 @ x
    a1 = np.maximum(0, z1)       # ReLU
    z2 = W2 @ a1
    y = np.maximum(0, z2)        # ReLU

    # Backward: dy/dx = diag(z2>0) @ W2 @ diag(z1>0) @ W1
    grad = np.diag((z2 > 0).astype(float)) @ W2 @ np.diag((z1 > 0).astype(float)) @ W1
    return y, grad

def gradient_through_residual_block(x, W1, W2):
    """Forward + backward through a residual block: y = relu(W2 @ relu(W1 @ x)) + x"""
    z1 = W1 @ x
    a1 = np.maximum(0, z1)
    z2 = W2 @ a1
    F_x = np.maximum(0, z2)
    y = F_x + x                  # Skip connection!

    # Backward: dy/dx = dF/dx + I
    dF_dx = np.diag((z2 > 0).astype(float)) @ W2 @ np.diag((z1 > 0).astype(float)) @ W1
    grad = dF_dx + np.eye(len(x))  # The +I that saves deep networks
    return y, grad

# Example: 8-dimensional vectors, random weights (small)
rng = np.random.RandomState(42)
d = 8
x = rng.randn(d)
W1 = rng.randn(d, d) * 0.3
W2 = rng.randn(d, d) * 0.3

_, grad_plain = gradient_through_plain_block(x, W1, W2)
_, grad_resid = gradient_through_residual_block(x, W1, W2)

print(f"Plain block  gradient norm: {np.linalg.norm(grad_plain):.4f}")
print(f"Residual block gradient norm: {np.linalg.norm(grad_resid):.4f}")
# Plain:    ~0.35  (shrinking through multiplications)
# Residual: ~3.12  (anchored near identity)
