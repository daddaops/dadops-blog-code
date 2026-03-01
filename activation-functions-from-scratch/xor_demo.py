"""
XOR demo: why activation functions matter.

Shows that a 2-layer network without nonlinear activation functions
cannot learn XOR, while the same architecture with sigmoid activations
learns it perfectly.

Requires: numpy

From: https://dadops.dev/blog/activation-functions-from-scratch/
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

def train_xor(use_activation, epochs=5000, lr=0.5):
    W1 = np.random.randn(2, 4) * 0.5
    b1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    b2 = np.zeros((1, 1))

    for _ in range(epochs):
        # Forward
        z1 = X @ W1 + b1
        a1 = sigmoid(z1) if use_activation else z1
        z2 = a1 @ W2 + b2
        out = sigmoid(z2)

        # Backward (manual gradients)
        err = out - y
        d2 = err * out * (1 - out)
        if use_activation:
            d1 = (d2 @ W2.T) * a1 * (1 - a1)
        else:
            d1 = d2 @ W2.T

        W2 -= lr * a1.T @ d2
        b2 -= lr * d2.sum(axis=0, keepdims=True)
        W1 -= lr * X.T @ d1
        b1 -= lr * d1.sum(axis=0, keepdims=True)

    z1 = X @ W1 + b1
    a1 = sigmoid(z1) if use_activation else z1
    return sigmoid(a1 @ W2 + b2)

print("WITHOUT activation:")
print(np.round(train_xor(False), 2))
# [[0.5], [0.5], [0.5], [0.5]]  <-- can't learn XOR

print("\nWITH activation:")
print(np.round(train_xor(True), 2))
# [[0.01], [0.98], [0.98], [0.03]]  <-- nails it
