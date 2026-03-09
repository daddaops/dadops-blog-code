"""Analytic infinite-width NTK via arccosine kernel recursion.

Uses Cho & Saul (2009) formula for ReLU networks to compute
the exact NTK without instantiating any weights.
"""
import numpy as np


def analytic_ntk_relu(X, depth):
    """Compute the infinite-width NTK for a depth-L ReLU network.

    Uses the arccosine kernel recursion (Cho & Saul 2009).
    """
    n = X.shape[0]

    # Base kernel: K^0 = X @ X^T / d_in
    K = X @ X.T / X.shape[1]

    # Arccosine kernels for ReLU
    def kappa1(K_diag_i, K_diag_j, K_ij):
        """Expected ReLU covariance: E[relu(u)*relu(v)]."""
        norms = np.sqrt(np.clip(K_diag_i * K_diag_j, 1e-12, None))
        cos_theta = np.clip(K_ij / norms, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return (1.0 / (2.0 * np.pi)) * norms * (np.sin(theta) + (np.pi - theta) * cos_theta)

    def kappa0(K_diag_i, K_diag_j, K_ij):
        """Expected ReLU derivative covariance: E[relu'(u)*relu'(v)]."""
        norms = np.sqrt(np.clip(K_diag_i * K_diag_j, 1e-12, None))
        cos_theta = np.clip(K_ij / norms, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        return (1.0 / (2.0 * np.pi)) * (np.pi - theta)

    di = np.diag(K)[:, None] * np.ones((1, n))
    dj = di.T

    # Build NTK layer by layer
    ntk = np.copy(K)  # contribution from first layer
    for l in range(1, depth + 1):
        # Derivative kernel for this layer
        K_dot = kappa0(di, dj, K)
        # Update NTK: previous layers' contribution gets multiplied by K_dot
        ntk = ntk * K_dot
        # Update NNGP kernel
        K = kappa1(di, dj, K)
        di = np.diag(K)[:, None] * np.ones((1, n))
        dj = di.T
        # Add this layer's own contribution
        ntk = ntk + K

    return ntk


if __name__ == "__main__":
    X = np.linspace(-1.5, 1.5, 20).reshape(-1, 1)

    for depth in [1, 2, 5]:
        analytic = analytic_ntk_relu(X, depth)
        diag_ratio = np.mean(np.diag(analytic)) / np.mean(np.abs(analytic))
        print(f"Depth {depth}: diag/mean ratio = {diag_ratio:.3f} "
              f"(higher = more local)")
