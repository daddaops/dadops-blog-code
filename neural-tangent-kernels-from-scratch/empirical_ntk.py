"""Empirical NTK computation via the Jacobian.

Builds J = df/d(params) and computes the NTK Gram matrix Theta = J @ J^T.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def build_network(d_in, width, rng):
    """Single hidden layer: f(x) = (1/sqrt(m)) * a^T relu(Wx)"""
    W = rng.randn(width, d_in) * np.sqrt(2.0 / d_in)  # He init
    a = rng.randn(width) * np.sqrt(1.0 / width)
    return W, a

def predict(x, W, a):
    """Forward pass for inputs x (n, d)."""
    m = W.shape[0]
    h = relu(x @ W.T)                   # (n, m)
    return h @ a / np.sqrt(m)            # (n,)

def empirical_ntk(x, W, a):
    """Compute NTK = J @ J^T where J = df/d(all params)."""
    n, d = x.shape
    m = W.shape[0]
    pre = x @ W.T                        # (n, m)  pre-activations
    h = relu(pre)                        # (n, m)
    mask = relu_deriv(pre)               # (n, m)  indicator for active ReLUs

    # Jacobian w.r.t. output weights a_j:  df/da_j = h_j / sqrt(m)
    J_a = h / np.sqrt(m)                 # (n, m)

    # Jacobian w.r.t. hidden weights W[j,:]:
    #   df/dW[j,k] = a_j * mask_j * x_k / sqrt(m)
    # Reshape for outer product across input dims
    J_W = (a[None, :] * mask)[:, :, None] * x[:, None, :] / np.sqrt(m)
    J_W = J_W.reshape(n, -1)             # (n, m*d)

    J = np.hstack([J_a, J_W])            # (n, m + m*d) full Jacobian
    return J @ J.T                        # (n, n) NTK Gram matrix


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    X = np.linspace(-2, 2, 20).reshape(-1, 1)
    W, a = build_network(1, 100, rng)

    ntk = empirical_ntk(X, W, a)
    print(f"NTK shape: {ntk.shape}")
    print(f"NTK[0,0] = {ntk[0,0]:.4f}")
    print(f"NTK[0,10] = {ntk[0,10]:.4f}")
    print(f"NTK is PSD: {np.all(np.linalg.eigvalsh(ntk) >= -1e-10)}")
