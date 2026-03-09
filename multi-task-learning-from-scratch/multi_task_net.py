"""Two-task neural network with shared hidden layers and separate output heads.

Demonstrates the hard-parameter sharing architecture: a shared encoder
feeds into task-specific regression and classification heads.
"""
import numpy as np

class MultiTaskNet:
    """Two-task network: shared hidden layers, separate output heads."""
    def __init__(self, d_in, d_hidden, seed=42):
        rng = np.random.RandomState(seed)
        scale = lambda fan_in: np.sqrt(2.0 / fan_in)
        # Shared layers
        self.W1 = rng.randn(d_in, d_hidden) * scale(d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.randn(d_hidden, d_hidden) * scale(d_hidden)
        self.b2 = np.zeros(d_hidden)
        # Regression head (1 output)
        self.W_reg = rng.randn(d_hidden, 1) * scale(d_hidden)
        self.b_reg = np.zeros(1)
        # Classification head (1 output, sigmoid)
        self.W_cls = rng.randn(d_hidden, 1) * scale(d_hidden)
        self.b_cls = np.zeros(1)

    def forward(self, X):
        # Shared encoder
        h1 = np.maximum(0, X @ self.W1 + self.b1)       # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)      # ReLU
        # Task-specific heads branch from shared representation
        y_reg = h2 @ self.W_reg + self.b_reg             # linear
        logits = h2 @ self.W_cls + self.b_cls
        y_cls = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        return y_reg.ravel(), y_cls.ravel(), h2


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 8)
    model = MultiTaskNet(d_in=8, d_hidden=32)
    y_reg, y_cls, h2 = model.forward(X)

    print(f"Input shape: {X.shape}")
    print(f"Shared repr shape: {h2.shape}")
    print(f"Regression output shape: {y_reg.shape}, mean={np.mean(y_reg):.4f}")
    print(f"Classification output shape: {y_cls.shape}, mean={np.mean(y_cls):.4f}")
    print(f"Classification range: [{np.min(y_cls):.4f}, {np.max(y_cls):.4f}]")

    # Count parameters
    shared = model.W1.size + model.b1.size + model.W2.size + model.b2.size
    reg_head = model.W_reg.size + model.b_reg.size
    cls_head = model.W_cls.size + model.b_cls.size
    print(f"\nShared params: {shared}")
    print(f"Regression head: {reg_head}")
    print(f"Classification head: {cls_head}")
    print(f"Total: {shared + reg_head + cls_head}")
