import numpy as np

class KANLayer:
    """A single KAN layer with B-spline activations on every edge."""
    def __init__(self, n_in, n_out, G=5, k=3):
        self.n_in, self.n_out, self.G, self.k = n_in, n_out, G, k
        self.n_basis = G + k
        # Augmented knot vector for each edge
        interior = np.linspace(-1, 1, G + 1)
        self.knots = np.concatenate([np.full(k, -1), interior, np.full(k, 1)])
        # Learnable spline coefficients: shape (n_in, n_out, n_basis)
        self.coeffs = np.random.randn(n_in, n_out, self.n_basis) * 0.1
        # Residual weights
        self.w_base = np.random.randn(n_in, n_out) * 0.1
        self.w_spline = np.ones((n_in, n_out))

    def _eval_basis(self, x):
        """Evaluate all B-spline basis functions at points x."""
        basis = np.zeros((len(x), self.n_basis))
        for h in range(self.n_basis):
            basis[:, h] = self._bspline(x, h, self.k)
        return basis  # shape (batch, n_basis)

    def _bspline(self, x, i, k):
        if k == 0:
            return np.where((self.knots[i] <= x) & (x < self.knots[i+1]), 1.0, 0.0)
        d1 = self.knots[i+k] - self.knots[i]
        d2 = self.knots[i+k+1] - self.knots[i+1]
        t1 = 0.0 if d1 == 0 else (x - self.knots[i]) / d1 * self._bspline(x, i, k-1)
        t2 = 0.0 if d2 == 0 else (self.knots[i+k+1] - x) / d2 * self._bspline(x, i+1, k-1)
        return t1 + t2

    def forward(self, x):
        """x: shape (batch, n_in) -> output: shape (batch, n_out)"""
        batch = x.shape[0]
        out = np.zeros((batch, self.n_out))
        for i in range(self.n_in):
            basis = self._eval_basis(x[:, i])           # (batch, n_basis)
            silu = x[:, i:i+1] / (1 + np.exp(-x[:, i:i+1]))  # SiLU base
            for j in range(self.n_out):
                spline_val = basis @ self.coeffs[i, j]  # (batch,)
                out[:, j] += self.w_base[i, j] * silu[:, 0] + self.w_spline[i, j] * spline_val
        return out

# Build a [2, 5, 1] KAN
layer1 = KANLayer(2, 5, G=5, k=3)
layer2 = KANLayer(5, 1, G=5, k=3)

x_sample = np.random.randn(10, 2)
h = layer1.forward(x_sample)   # (10, 5)
y = layer2.forward(h)          # (10, 1)

kan_params = 2*5*(5+3) + 5*1*(5+3) + 2*5 + 5*1 + 2*5 + 5*1  # coeffs + w_base + w_spline
mlp_params = 2*5 + 5 + 5*1 + 1  # weights + biases for equivalent MLP
print(f"KAN [2,5,1] params: {kan_params}, MLP [2,5,1] params: {mlp_params}")
# KAN [2,5,1] params: 150, MLP [2,5,1] params: 21
