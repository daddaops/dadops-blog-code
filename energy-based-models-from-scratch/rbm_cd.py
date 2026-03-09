"""
Restricted Boltzmann Machine with Contrastive Divergence

Implements an RBM trained with CD-1 on small binary patterns.
Demonstrates the positive/negative phase weight update.

Blog post: https://dadops.dev/blog/energy-based-models-from-scratch/
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class RBM:
    def __init__(self, n_visible, n_hidden, lr=0.1):
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.b = np.zeros(n_visible)   # visible bias
        self.c = np.zeros(n_hidden)    # hidden bias
        self.lr = lr

    def sample_hidden(self, v):
        """p(h=1|v) = sigmoid(v @ W + c)"""
        prob_h = sigmoid(v @ self.W + self.c)
        return prob_h, (np.random.rand(*prob_h.shape) < prob_h).astype(float)

    def sample_visible(self, h):
        """p(v=1|h) = sigmoid(h @ W.T + b)"""
        prob_v = sigmoid(h @ self.W.T + self.b)
        return prob_v, (np.random.rand(*prob_v.shape) < prob_v).astype(float)

    def contrastive_divergence(self, v_data, k=1):
        """CD-k: k steps of Gibbs sampling from data."""
        # Positive phase: clamp visible to data
        prob_h0, h0 = self.sample_hidden(v_data)

        # Negative phase: k Gibbs steps
        h_k = h0
        for _ in range(k):
            prob_v_k, v_k = self.sample_visible(h_k)
            prob_h_k, h_k = self.sample_hidden(v_k)

        # Weight update: push energy down on data, up on reconstructions
        batch_size = v_data.shape[0]
        self.W += self.lr * (v_data.T @ prob_h0 - v_k.T @ prob_h_k) / batch_size
        self.b += self.lr * np.mean(v_data - v_k, axis=0)
        self.c += self.lr * np.mean(prob_h0 - prob_h_k, axis=0)
        return np.mean((v_data - prob_v_k) ** 2)  # reconstruction error

# Train on small binary patterns
np.random.seed(42)
patterns = np.array([
    [1,1,0,0,0,0],  [1,1,0,0,0,0],  [0,0,0,0,1,1],
    [0,0,0,0,1,1],  [1,0,1,0,1,0],  [1,0,1,0,1,0],
], dtype=float)

rbm = RBM(n_visible=6, n_hidden=4, lr=0.1)
for epoch in range(500):
    err = rbm.contrastive_divergence(patterns, k=1)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: reconstruction error = {err:.4f}")
# Epoch 0: 0.2847 → Epoch 400: 0.0312
