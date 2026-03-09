"""Overfitting demonstration — sine wave fitting with no regularization."""
import numpy as np
np.random.seed(42)

# Generate 50 noisy sine points (train) and 200 clean points (test)
X_train = np.random.uniform(-3, 3, (50, 1))
y_train = np.sin(X_train) + np.random.randn(50, 1) * 0.3
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_test = np.sin(X_test)

# 4-layer ReLU network: 1 -> 64 -> 64 -> 64 -> 1
dims = [1, 64, 64, 64, 1]
W = [np.random.randn(dims[i], dims[i+1]) * 0.5 for i in range(4)]
b = [np.zeros((1, dims[i+1])) for i in range(4)]
lr = 0.005

for epoch in range(3000):
    # Forward pass
    h = X_train
    activations = [h]
    for i in range(3):
        h = h @ W[i] + b[i]
        h = np.maximum(0, h)  # ReLU
        activations.append(h)
    out = h @ W[3] + b[3]

    # MSE loss + backprop
    loss = np.mean((out - y_train) ** 2)
    grad = 2 * (out - y_train) / len(y_train)
    for i in range(3, -1, -1):
        W[i] -= lr * activations[i].T @ grad
        b[i] -= lr * grad.sum(axis=0, keepdims=True)
        if i > 0:
            grad = (grad @ W[i].T) * (activations[i] > 0)

# Evaluate on test set
h = X_test
for i in range(3):
    h = np.maximum(0, h @ W[i] + b[i])
test_pred = h @ W[3] + b[3]
test_loss = np.mean((test_pred - y_test) ** 2)

print(f"Train loss: {loss:.4f}")
print(f"Test loss:  {test_loss:.4f}")
print(f"Gap:        {test_loss - loss:.4f} — the network memorized, not learned")
