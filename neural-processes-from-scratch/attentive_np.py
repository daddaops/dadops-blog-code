"""Attentive Neural Process with cross-attention.

Replaces mean pooling with query-specific attention so each
target point focuses on the most relevant context points.
"""
import numpy as np
from cnp import make_sine_task


class AttentiveNP:
    def __init__(self, h_dim=64, n_heads=4):
        self.h_dim = h_dim
        self.d_k = h_dim // n_heads
        self.n_heads = n_heads
        # Encoder
        self.enc_W1 = np.random.randn(2, h_dim) * 0.1
        self.enc_b1 = np.zeros(h_dim)
        # Attention projections (per-head)
        self.W_Q = np.random.randn(1, h_dim) * 0.1   # query from x*
        self.W_K = np.random.randn(h_dim, h_dim) * 0.1  # key from r_i
        self.W_V = np.random.randn(h_dim, h_dim) * 0.1  # value from r_i
        # Decoder
        self.dec_W1 = np.random.randn(h_dim + 1, h_dim) * 0.1
        self.dec_b1 = np.zeros(h_dim)
        self.dec_W2 = np.random.randn(h_dim, 2) * 0.1
        self.dec_b2 = np.zeros(2)

    def cross_attend(self, x_target, ctx_repr):
        """Each target attends to context representations."""
        Q = x_target @ self.W_Q          # (m, h_dim)
        K = ctx_repr @ self.W_K           # (n, h_dim)
        V = ctx_repr @ self.W_V           # (n, h_dim)
        # Scaled dot-product attention
        scores = Q @ K.T / np.sqrt(self.d_k)  # (m, n)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)  # softmax
        return weights @ V                 # (m, h_dim)

    def predict(self, x_ctx, y_ctx, x_target):
        pairs = np.column_stack([x_ctx, y_ctx])
        ctx_repr = np.maximum(0, pairs @ self.enc_W1 + self.enc_b1)
        attended = self.cross_attend(x_target, ctx_repr)  # (m, h_dim)
        inp = np.column_stack([attended, x_target])
        h = np.maximum(0, inp @ self.dec_W1 + self.dec_b1)
        out = h @ self.dec_W2 + self.dec_b2
        return out[:, 0], np.exp(out[:, 1]) + 1e-4


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x_all, y_all = make_sine_task(rng)
    x_ctx, y_ctx = x_all[:5], y_all[:5]
    x_tgt = x_all[5:]

    anp = AttentiveNP()
    mu, sigma = anp.predict(x_ctx, y_ctx, x_tgt)
    print(f"ANP predictions at first 3 targets:")
    for i in range(3):
        print(f"  x={x_tgt[i,0]:.2f}  pred={mu[i]:.2f} +/- {sigma[i]:.2f}")
