"""Rich vs lazy training regimes.

Measures how much the NTK changes during training at different widths.
Narrow networks: NTK changes a lot (rich/feature-learning regime).
Wide networks: NTK barely changes (lazy/NTK regime).
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def compute_ntk(X, W, a):
    """Compute empirical NTK for 1-hidden-layer ReLU network."""
    n, d = X.shape
    m = W.shape[0]
    pre = X @ W.T
    h, mask = relu(pre), relu_deriv(pre)
    J_a = h / np.sqrt(m)
    J_W = (a[None, :] * mask)[:, :, None] * X[:, None, :] / np.sqrt(m)
    J = np.hstack([J_a, J_W.reshape(n, -1)])
    return J @ J.T

def train_network(X, Y, width, lr, steps, rng):
    """Train and return NTK change ratio."""
    n, d = X.shape
    m = width
    W = rng.randn(m, d) * np.sqrt(2.0 / d)
    a = rng.randn(m) * np.sqrt(1.0 / m)
    ntk_init = compute_ntk(X, W, a)

    for t in range(steps):
        pre = X @ W.T
        h = relu(pre)
        f = h @ a / np.sqrt(m)
        r = f - Y
        da = h.T @ r / np.sqrt(m)
        dW = np.zeros_like(W)
        for i in range(n):
            mask_i = relu_deriv(pre[i])
            dW += r[i] * (a * mask_i)[:, None] * X[i][None, :] / np.sqrt(m)
        a -= lr * da
        W -= lr * dW

    ntk_final = compute_ntk(X, W, a)
    change = np.linalg.norm(ntk_final - ntk_init) / np.linalg.norm(ntk_init)
    return change


if __name__ == "__main__":
    X = np.linspace(-2, 2, 15).reshape(-1, 1)
    Y = np.sin(2 * X.ravel())
    widths = [20, 50, 100, 500, 2000]

    for m in widths:
        change = train_network(X, Y, m, lr=0.5/m, steps=300, rng=np.random.RandomState(42))
        regime = "RICH (feature learning)" if change > 0.1 else "LAZY (NTK regime)"
        print(f"Width {m:>4d}: NTK change = {change:.4f}  [{regime}]")
