"""Conditional Neural Process (CNP) from scratch.

Encoder maps (x,y) context pairs to a fixed-size representation
via mean pooling. Decoder predicts mu and sigma at target locations.
"""
import numpy as np


class CNP:
    def __init__(self, h_dim=64):
        # Encoder: (x, y) -> r_i
        self.W1 = np.random.randn(2, h_dim) * 0.1
        self.b1 = np.zeros(h_dim)
        self.W2 = np.random.randn(h_dim, h_dim) * 0.1
        self.b2 = np.zeros(h_dim)
        # Decoder: (r, x*) -> (mu, log_sigma)
        self.W3 = np.random.randn(h_dim + 1, h_dim) * 0.1
        self.b3 = np.zeros(h_dim)
        self.W4 = np.random.randn(h_dim, 2) * 0.1
        self.b4 = np.zeros(2)

    def encode(self, x_ctx, y_ctx):
        """Encode each context pair, then mean-pool."""
        pairs = np.column_stack([x_ctx, y_ctx])        # (n, 2)
        h = np.maximum(0, pairs @ self.W1 + self.b1)   # ReLU
        r_all = np.maximum(0, h @ self.W2 + self.b2)   # (n, h_dim)
        return r_all.mean(axis=0)                       # mean pool

    def decode(self, r, x_target):
        """Predict mu, sigma at each target x."""
        inp = np.column_stack([np.tile(r, (len(x_target), 1)),
                               x_target])               # (m, h_dim+1)
        h = np.maximum(0, inp @ self.W3 + self.b3)
        out = h @ self.W4 + self.b4                      # (m, 2)
        mu = out[:, 0]
        sigma = np.exp(out[:, 1]) + 1e-4                 # positive via exp
        return mu, sigma

    def predict(self, x_ctx, y_ctx, x_target):
        r = self.encode(x_ctx, y_ctx)
        return self.decode(r, x_target)


def make_sine_task(rng):
    """Generate a random sine task: y = A*sin(x + phase)."""
    A, phase = rng.uniform(0.5, 2.0), rng.uniform(0, 2 * np.pi)
    x_all = rng.uniform(-3, 3, size=(20, 1))
    y_all = A * np.sin(x_all + phase)
    return x_all, y_all


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x_all, y_all = make_sine_task(rng)

    # Split into 5 context + 15 target
    x_ctx, y_ctx = x_all[:5], y_all[:5]
    x_tgt, y_tgt = x_all[5:], y_all[5:]

    cnp = CNP()
    mu, sigma = cnp.predict(x_ctx, y_ctx, x_tgt)
    print(f"Predictions at first 3 targets:")
    for i in range(3):
        print(f"  x={x_tgt[i,0]:.2f}  true={y_tgt[i,0]:.2f}"
              f"  pred={mu[i]:.2f} +/- {sigma[i]:.2f}")
