"""MDN architecture: forward pass producing mixture parameters.

Demonstrates the Mixture Density Network class with softmax mixing
weights, identity means, and exp-transformed standard deviations.
"""
import numpy as np

class MDN:
    """Mixture Density Network: 1 input, 2 hidden layers, K mixture components."""
    def __init__(self, n_hidden=64, K=3):
        self.K = K
        # Xavier initialization for hidden layers
        self.W1 = np.random.randn(1, n_hidden) * np.sqrt(2.0 / 1)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_hidden)
        # Output layer: 3K outputs (K pis + K mus + K log_sigmas)
        self.W3 = np.random.randn(n_hidden, 3 * K) * np.sqrt(2.0 / n_hidden)
        self.b3 = np.zeros(3 * K)

    def forward(self, x):
        """Forward pass returning (pi, mu, sigma) for each component."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)            # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)           # ReLU
        out = h2 @ self.W3 + self.b3                          # (N, 3K)

        # Split and apply activations
        z_pi  = out[:, :self.K]                               # mixing logits
        z_mu  = out[:, self.K:2*self.K]                       # means (identity)
        z_sig = out[:, 2*self.K:]                             # log-std devs

        # Softmax for mixing coefficients
        z_pi_shifted = z_pi - z_pi.max(axis=1, keepdims=True)
        exp_pi = np.exp(z_pi_shifted)
        pi = exp_pi / exp_pi.sum(axis=1, keepdims=True)

        mu = z_mu
        sigma = np.exp(np.clip(z_sig, -7, 7))                # exp with clamp

        return pi, mu, sigma

mdn = MDN(n_hidden=64, K=3)
x_test = np.array([[0.5]])
pi, mu, sigma = mdn.forward(x_test)
print(f"Mixing weights: {pi[0]}")     # e.g., [0.35, 0.41, 0.24]
print(f"Means:          {mu[0]}")      # e.g., [-0.12, 0.53, 0.21]
print(f"Std devs:       {sigma[0]}")   # e.g., [0.89, 1.02, 0.67]
