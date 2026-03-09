import numpy as np

class FourierKANLayer:
    """KAN layer using Fourier series instead of B-splines."""
    def __init__(self, n_in, n_out, n_freq=5):
        self.n_in, self.n_out, self.n_freq = n_in, n_out, n_freq
        # 2 * n_freq coefficients per edge (cos + sin at each frequency)
        self.a = np.random.randn(n_in, n_out, n_freq) * 0.1  # cosine coeffs
        self.b = np.random.randn(n_in, n_out, n_freq) * 0.1  # sine coeffs

    def forward(self, x):
        """x: (batch, n_in) -> (batch, n_out)"""
        batch = x.shape[0]
        out = np.zeros((batch, self.n_out))
        for i in range(self.n_in):
            for j in range(self.n_out):
                val = np.zeros(batch)
                for k in range(self.n_freq):
                    val += self.a[i, j, k] * np.cos((k+1) * x[:, i])
                    val += self.b[i, j, k] * np.sin((k+1) * x[:, i])
                out[:, j] += val
        return out

    @property
    def n_params(self):
        return self.n_in * self.n_out * 2 * self.n_freq

# Compare on a periodic function: sin(3x) + cos(2y)
np.random.seed(42)
X = np.random.uniform(-np.pi, np.pi, (200, 2))
y_true = np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])

fourier_layer = FourierKANLayer(2, 1, n_freq=5)  # 20 params
print(f"FourierKAN params: {fourier_layer.n_params}")
# Fourier basis is a natural fit for periodic targets --
# frequency 3 (sin) and frequency 2 (cos) are directly representable
# B-spline KAN would need many grid points to approximate these oscillations
