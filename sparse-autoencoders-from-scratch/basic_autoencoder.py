"""Basic autoencoder concept: demonstrating superposition in neural networks.

5 sparse features packed into 2 dimensions, showing how polysemantic neurons arise.

From: https://dadops.substack.com/sparse-autoencoders-from-scratch
"""
import numpy as np


def demonstrate_superposition():
    """5 sparse features packed into 2 dimensions."""
    np.random.seed(42)
    d_features, d_hidden = 5, 2

    # Ground-truth features: 5 one-hot vectors in 5D
    # Each data point activates 1-2 features (sparse!)
    n_samples = 500
    feature_probs = 0.15  # each feature active 15% of the time
    X = np.random.binomial(1, feature_probs, (n_samples, d_features)).astype(float)
    X *= np.random.uniform(0.5, 2.0, X.shape)  # random magnitudes

    # Importance weights: feature 0 most important, feature 4 least
    importance = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    # Train a linear model to reconstruct X through a 2D bottleneck
    # Tied weights: X_hat = X @ W @ W.T, minimize importance-weighted MSE
    W = np.random.randn(d_features, d_hidden) * 0.5
    lr = 0.001
    for step in range(2000):
        H = X @ W                      # encode: (n, 2)
        X_hat = H @ W.T               # decode: (n, 5)
        R = importance * (X_hat - X)   # weighted residual: (n, 5)
        # Gradient via both paths through W (encoding and decoding)
        grad = X.T @ (R @ W) / n_samples      # path 1: through encoder
        grad += (R.T @ H) / n_samples         # path 2: through decoder
        W -= lr * 2 * grad

    # The learned directions for each feature
    print("Feature directions in 2D space (rows of W):")
    for i in range(d_features):
        direction = W[i] / (np.linalg.norm(W[i]) + 1e-8)
        print(f"  Feature {i} (importance={importance[i]:.1f}): "
              f"[{direction[0]:+.3f}, {direction[1]:+.3f}]")

    # Check polysemanticity: each hidden neuron responds to multiple features
    print("\nHidden neuron 0 responds to features:", end=" ")
    print([i for i in range(d_features) if abs(W[i, 0]) > 0.2])
    print("Hidden neuron 1 responds to features:", end=" ")
    print([i for i in range(d_features) if abs(W[i, 1]) > 0.2])


if __name__ == "__main__":
    demonstrate_superposition()
