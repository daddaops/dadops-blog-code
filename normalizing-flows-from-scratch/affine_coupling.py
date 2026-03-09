"""Affine coupling layer (RealNVP).

Implements a single affine coupling layer with an MLP that outputs
scale and translation parameters. Demonstrates perfect invertibility.
"""
import numpy as np


class AffineCouplingLayer:
    """Single RealNVP-style affine coupling layer."""
    def __init__(self, d, hidden=32, seed=0):
        rng = np.random.RandomState(seed)
        self.d_half = d // 2
        # Simple MLP: input(d_half) -> hidden -> output(2 * d_half) for s and t
        self.W1 = rng.randn(self.d_half, hidden) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, self.d_half * 2) * 0.1
        self.b2 = np.zeros(self.d_half * 2)

    def _net(self, x1):
        """MLP that outputs scale (s) and translation (t)."""
        h = np.maximum(0, x1 @ self.W1 + self.b1)  # ReLU
        out = h @ self.W2 + self.b2
        s = out[:, :self.d_half]   # scale (log-scale)
        t = out[:, self.d_half:]   # translation
        return s, t

    def forward(self, z):
        z1, z2 = z[:, :self.d_half], z[:, self.d_half:]
        s, t = self._net(z1)
        x1 = z1
        x2 = z2 * np.exp(s) + t
        log_det = np.sum(s, axis=1)
        return np.hstack([x1, x2]), log_det

    def inverse(self, x):
        x1, x2 = x[:, :self.d_half], x[:, self.d_half:]
        s, t = self._net(x1)
        z1 = x1
        z2 = (x2 - t) * np.exp(-s)
        return np.hstack([z1, z2])


if __name__ == "__main__":
    # Verify invertibility: f_inv(f(z)) should equal z
    np.random.seed(42)
    layer = AffineCouplingLayer(d=4, hidden=16, seed=7)
    z_test = np.random.randn(100, 4)
    x_test, log_det = layer.forward(z_test)
    z_recovered = layer.inverse(x_test)

    reconstruction_error = np.max(np.abs(z_test - z_recovered))
    print(f"Max reconstruction error: {reconstruction_error:.2e}")
    print(f"Mean log|det J|: {log_det.mean():.4f}")
