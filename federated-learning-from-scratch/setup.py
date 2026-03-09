"""
Shared setup for federated learning scripts.

Creates 5 hospitals with 40 patients each, plus pooled baseline.

Blog post: https://dadops.dev/blog/federated-learning-from-scratch/
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def make_hospitals():
    """Create 5 hospitals with different patient populations."""
    np.random.seed(42)
    true_w = np.array([0.8, -1.2])

    hospitals = []
    for i in range(5):
        rng = np.random.RandomState(i)
        shift = rng.randn(2) * 1.5
        X = rng.randn(40, 2) + shift
        y = (sigmoid(X @ true_w) > rng.random(40)).astype(float)
        hospitals.append((X, y))

    X_all = np.vstack([h[0] for h in hospitals])
    y_all = np.concatenate([h[1] for h in hospitals])
    return hospitals, X_all, y_all, true_w


def train(X, y, lr=0.5, epochs=100):
    w = np.zeros(2)
    for _ in range(epochs):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        w -= lr * grad
    return w


def acc(X, y, w):
    return ((sigmoid(X @ w) > 0.5) == y).mean()
