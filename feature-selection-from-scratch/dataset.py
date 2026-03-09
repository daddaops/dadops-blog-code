"""
Shared dataset and utilities for feature selection scripts.

Generates a synthetic dataset with 3 informative, 3 redundant, and 4 noise features.

Blog post: https://dadops.dev/blog/feature-selection-from-scratch/
"""
import numpy as np


def make_dataset():
    """Generate the feature selection dataset. Returns X, y, names, n."""
    np.random.seed(42)
    n = 200

    # 3 informative features
    X_info = np.random.randn(n, 3)
    # Binary target: positive when weighted sum exceeds threshold
    w_true = np.array([1.5, -1.0, 0.8])
    y = (X_info @ w_true + 0.3 * np.random.randn(n) > 0).astype(int)

    # 3 redundant features (linear combos of informative + noise)
    X_redun = np.column_stack([
        0.9 * X_info[:, 0] + 0.1 * np.random.randn(n),
        0.8 * X_info[:, 1] + 0.2 * np.random.randn(n),
        0.7 * X_info[:, 2] + 0.3 * np.random.randn(n),
    ])
    # 4 pure noise features
    X_noise = np.random.randn(n, 4)
    X = np.column_stack([X_info, X_redun, X_noise])
    names = [f"info_{i}" for i in range(3)] + \
            [f"redun_{i}" for i in range(3)] + \
            [f"noise_{i}" for i in range(4)]

    return X, y, names, n


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))
