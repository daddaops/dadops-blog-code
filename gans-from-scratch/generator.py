import numpy as np

class Generator:
    """Maps random noise z to data space through a 2-layer MLP."""
    def __init__(self, noise_dim=1, hidden_dim=64, output_dim=1):
        # He initialization for ReLU networks
        self.W1 = np.random.randn(noise_dim, hidden_dim) * np.sqrt(2.0 / noise_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, z):
        """z -> hidden (ReLU) -> output (linear)."""
        self.z = z                                           # (batch, noise_dim)
        self.h = np.maximum(0, z @ self.W1 + self.b1)       # (batch, hidden_dim)
        self.out = self.h @ self.W2 + self.b2                # (batch, output_dim)
        return self.out

    def backward(self, grad_out, lr=0.001):
        """Backprop through generator and update weights."""
        # Gradient through output layer
        grad_W2 = self.h.T @ grad_out                       # (hidden, output)
        grad_b2 = grad_out.sum(axis=0)
        grad_h = grad_out @ self.W2.T                        # (batch, hidden)
        # Gradient through ReLU
        grad_h = grad_h * (self.h > 0)                      # mask dead neurons
        # Gradient through first layer
        grad_W1 = self.z.T @ grad_h                          # (noise, hidden)
        grad_b1 = grad_h.sum(axis=0)
        # SGD update
        self.W2 -= lr * grad_W2 / len(grad_out)
        self.b2 -= lr * grad_b2 / len(grad_out)
        self.W1 -= lr * grad_W1 / len(grad_out)
        self.b1 -= lr * grad_b1 / len(grad_out)
