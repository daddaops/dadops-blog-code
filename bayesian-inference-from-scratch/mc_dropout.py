"""MC Dropout Uncertainty from Scratch — neural net uncertainty estimation.

Demonstrates how running multiple forward passes with dropout at inference
time produces varying uncertainty across the input space.
"""
import numpy as np

class SimpleNeuralNet:
    """2-layer neural net with dropout for MC uncertainty estimation."""
    def __init__(self, input_dim, hidden_dim, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * 0.5
        self.b2 = np.zeros(1)

    def forward(self, X, dropout_rate=0.0):
        """Forward pass with optional dropout."""
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        if dropout_rate > 0:
            mask = np.random.binomial(1, 1 - dropout_rate, h.shape)
            h = h * mask / (1 - dropout_rate)      # inverted dropout
        return h @ self.W2 + self.b2

    def mc_predict(self, X, n_forward=50, dropout_rate=0.3):
        """MC Dropout: run multiple forward passes, measure variance."""
        predictions = np.array([self.forward(X, dropout_rate) for _ in range(n_forward)])
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        return mean_pred.flatten(), std_pred.flatten()

# Demo: uncertainty should be high where there's no training data
net = SimpleNeuralNet(1, 32)
x_test = np.linspace(-5, 5, 100).reshape(-1, 1)
mean_pred, std_pred = net.mc_predict(x_test, n_forward=100, dropout_rate=0.3)

# Show uncertainty varies across the input space
for region, mask in [("center", np.abs(x_test.flatten()) < 1),
                     ("edges", np.abs(x_test.flatten()) > 3)]:
    print(f"Avg uncertainty ({region}): {std_pred[mask].mean():.3f}")
