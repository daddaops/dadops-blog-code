"""Planar flow layers.

Applies simple shear transformations with computable Jacobian
determinants. Stacks 8 layers to transform a 2D Gaussian.
"""
import numpy as np


def planar_flow(z, w, u, b):
    """Single planar flow layer: f(z) = z + u * tanh(w^T z + b)."""
    activation = np.tanh(z @ w + b)  # (N,)
    return z + np.outer(activation, u)  # (N, d)

def planar_log_det(z, w, u, b):
    """Log |det Jacobian| for planar flow."""
    h_prime = 1 - np.tanh(z @ w + b)**2  # derivative of tanh
    psi = np.outer(h_prime, w)            # (N, d)
    det_term = np.abs(1 + psi @ u)        # (N,)
    return np.log(det_term + 1e-10)


if __name__ == "__main__":
    # Stack 8 planar flow layers to transform a 2D Gaussian
    np.random.seed(42)
    d = 2
    z = np.random.normal(0, 1, (2000, d))
    log_prob = -0.5 * np.sum(z**2, axis=1) - d/2 * np.log(2 * np.pi)

    for k in range(8):
        rng = np.random.RandomState(k + 10)
        w = rng.randn(d) * 1.5
        u = rng.randn(d) * 0.8
        b = rng.randn() * 0.5
        log_prob += planar_log_det(z, w, u, b)
        z = planar_flow(z, w, u, b)

    print(f"Input range:  x=[{z[:,0].min():.1f}, {z[:,0].max():.1f}], "
          f"y=[{z[:,1].min():.1f}, {z[:,1].max():.1f}]")
    print(f"Mean log p(x): {log_prob.mean():.2f}")
