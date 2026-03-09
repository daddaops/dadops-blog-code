"""Shared utilities for continual learning scripts."""
import numpy as np


def make_task(center_a, center_b, n=100, seed=42):
    rng = np.random.RandomState(seed)
    X = np.vstack([rng.randn(n, 2) * 0.5 + center_a,
                   rng.randn(n, 2) * 0.5 + center_b])
    y = np.array([0]*n + [1]*n)
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_mlp(X, y, W1, b1, W2, b2, epochs=200, lr=0.05):
    for _ in range(epochs):
        h = np.maximum(0, X @ W1 + b1)          # ReLU hidden
        out = sigmoid(h @ W2 + b2)               # sigmoid output
        err = out.ravel() - y
        dW2 = h.T @ err.reshape(-1, 1) / len(y)
        db2 = err.mean()
        dh = err.reshape(-1, 1) * W2.T * (h > 0) # ReLU grad
        dW1 = X.T @ dh / len(y)
        db1 = dh.mean(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    return W1, b1, W2, b2


def accuracy(X, y, W1, b1, W2, b2):
    h = np.maximum(0, X @ W1 + b1)
    pred = (sigmoid(h @ W2 + b2).ravel() > 0.5).astype(int)
    return (pred == y).mean()


def init_weights(seed=0, n_in=2, n_hid=8):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(n_in, n_hid) * 0.3
    b1 = np.zeros(n_hid)
    W2 = rng.randn(n_hid, 1) * 0.3
    b2 = np.zeros(1)
    return W1, b1, W2, b2
