"""
Model-Wise Double Descent with Neural Networks

Trains 2-layer ReLU MLPs of increasing width on a small noisy dataset,
demonstrating the double descent pattern in neural networks.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


def train_mlp(X, y, hidden, lr=0.01, epochs=2000):
    """Train a 2-layer ReLU MLP: input -> hidden -> 1 output."""
    np.random.seed(7)
    d = X.shape[1]
    W1 = np.random.randn(d, hidden) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden)
    b2 = np.zeros(1)

    for _ in range(epochs):
        # Forward
        h = relu(X @ W1 + b1)
        pred = h @ W2 + b2

        # MSE loss gradient
        err = pred - y.reshape(-1, 1)
        grad_W2 = h.T @ err / len(X)
        grad_b2 = err.mean(axis=0)
        grad_h = err @ W2.T
        grad_h[h == 0] = 0  # ReLU backward
        grad_W1 = X.T @ grad_h / len(X)
        grad_b1 = grad_h.mean(axis=0)

        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1

    return W1, b1, W2, b2


# Generate a small noisy dataset
np.random.seed(42)
n = 40
X_train = np.random.randn(n, 5)
true_w = np.array([1, -0.5, 0.3, 0, 0.8])
y_train = X_train @ true_w + np.random.randn(n) * 0.5
X_test = np.random.randn(200, 5)
y_test = X_test @ true_w

widths = [3, 5, 10, 20, 40, 60, 100, 200, 500]
for h in widths:
    W1, b1, W2, b2 = train_mlp(X_train, y_train, h)
    pred_tr = relu(X_train @ W1 + b1) @ W2 + b2
    pred_te = relu(X_test @ W1 + b1) @ W2 + b2
    tr_err = np.mean((pred_tr.flatten() - y_train) ** 2)
    te_err = np.mean((pred_te.flatten() - y_test) ** 2)
    params = 5 * h + h + h + 1  # W1 + b1 + W2 + b2
    print(f"width={h:>3d} params={params:>4d} (p/n={params/n:.1f})  "
          f"train={tr_err:.4f}  test={te_err:.4f}")


if __name__ == "__main__":
    pass  # Output printed above
