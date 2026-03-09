import numpy as np

class Discriminator:
    """Classifies samples as real (1) or fake (0) via a 2-layer MLP."""
    def __init__(self, input_dim=1, hidden_dim=64):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    def forward(self, x):
        """x -> hidden (ReLU) -> probability (sigmoid)."""
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)       # (batch, hidden)
        logit = self.h @ self.W2 + self.b2                   # (batch, 1)
        self.prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))  # sigmoid
        return self.prob

    def backward(self, grad_prob, lr=0.001):
        """Backprop through discriminator and update weights."""
        # Gradient through sigmoid: dsigmoid/dlogit = prob * (1 - prob)
        grad_logit = grad_prob * self.prob * (1.0 - self.prob)
        # Output layer
        grad_W2 = self.h.T @ grad_logit
        grad_b2 = grad_logit.sum(axis=0)
        grad_h = grad_logit @ self.W2.T
        # ReLU
        grad_h = grad_h * (self.h > 0)
        # First layer
        grad_W1 = self.x.T @ grad_h
        grad_b1 = grad_h.sum(axis=0)
        # SGD update
        self.W2 -= lr * grad_W2 / len(grad_prob)
        self.b2 -= lr * grad_b2 / len(grad_prob)
        self.W1 -= lr * grad_W1 / len(grad_prob)
        self.b1 -= lr * grad_b1 / len(grad_prob)
        # Return gradient w.r.t. input (needed for generator training)
        grad_x = grad_h @ self.W1.T
        return grad_x
