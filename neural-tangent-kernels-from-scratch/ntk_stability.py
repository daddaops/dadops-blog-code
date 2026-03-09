"""NTK convergence with width.

Two independent random initializations produce increasingly
similar NTK matrices as width grows: O(1/sqrt(m)) variation.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def make_net_and_ntk(d_in, m, X, rng):
    """Build network, compute NTK, return (W, a, ntk_matrix)."""
    W = rng.randn(m, d_in) * np.sqrt(2.0 / d_in)
    a = rng.randn(m) * np.sqrt(1.0 / m)
    n = X.shape[0]
    pre = X @ W.T
    h = relu(pre)
    mask = relu_deriv(pre)
    J_a = h / np.sqrt(m)
    J_W = (a[None, :] * mask)[:, :, None] * X[:, None, :] / np.sqrt(m)
    J = np.hstack([J_a, J_W.reshape(n, -1)])
    return W, a, J @ J.T


if __name__ == "__main__":
    X = np.linspace(-2, 2, 15).reshape(-1, 1)
    widths = [10, 50, 200, 1000]

    for m in widths:
        rng1, rng2 = np.random.RandomState(0), np.random.RandomState(1)
        _, _, ntk1 = make_net_and_ntk(1, m, X, rng1)
        _, _, ntk2 = make_net_and_ntk(1, m, X, rng2)
        diff = np.linalg.norm(ntk1 - ntk2) / np.linalg.norm(ntk1)
        print(f"Width {m:>4d}: NTK variation between inits = {diff:.4f}")
