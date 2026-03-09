"""Regularization toolkit — modular implementations + comparison."""
import numpy as np

# Generate data with same seed as blog
np.random.seed(42)
X_train = np.random.uniform(-3, 3, (50, 1))
y_train = np.sin(X_train) + np.random.randn(50, 1) * 0.3
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_test = np.sin(X_test)

def dropout_mask(shape, p=0.3):
    """Inverted dropout: zero out with prob p, scale survivors by 1/(1-p)."""
    mask = (np.random.rand(*shape) > p).astype(float)
    return mask / (1 - p)

def train_network(use_l2=False, use_dropout=False, use_early_stop=False,
                  use_label_smooth=False, lam=0.01, drop_p=0.3,
                  patience=200, epsilon=0.1, epochs=3000):
    np.random.seed(42)
    dims = [1, 64, 64, 64, 1]
    W = [np.random.randn(dims[i], dims[i+1]) * 0.5 for i in range(4)]
    b = [np.zeros((1, dims[i+1])) for i in range(4)]
    best_val, wait, best_W, best_b = 1e9, 0, None, None

    for epoch in range(epochs):
        h = X_train
        acts, masks = [h], []
        for i in range(3):
            h = np.maximum(0, h @ W[i] + b[i])
            if use_dropout:
                m = dropout_mask(h.shape, drop_p)
                h = h * m
                masks.append(m)
            else:
                masks.append(np.ones_like(h))
            acts.append(h)
        out = h @ W[3] + b[3]

        target = y_train
        if use_label_smooth:
            target = (1 - epsilon) * y_train + epsilon * np.mean(y_train)
        grad = 2 * (out - target) / len(y_train)

        for i in range(3, -1, -1):
            dW = acts[i].T @ grad
            if use_l2:
                dW += 2 * lam * W[i]
            W[i] -= 0.005 * dW
            b[i] -= 0.005 * grad.sum(axis=0, keepdims=True)
            if i > 0:
                grad = (grad @ W[i].T) * (acts[i] > 0) * masks[i-1]

        if use_early_stop and epoch % 50 == 0:
            h = X_test
            for i in range(3):
                h = np.maximum(0, h @ W[i] + b[i])
            val = np.mean((h @ W[3] + b[3] - y_test) ** 2)
            if val < best_val:
                best_val = val
                wait = 0
                best_W = [w.copy() for w in W]
                best_b = [bi.copy() for bi in b]
            else:
                wait += 1
                if wait >= patience // 50:
                    W, b = best_W, best_b
                    break

    h = X_test
    for i in range(3):
        h = np.maximum(0, h @ W[i] + b[i])
    return np.mean((h @ W[3] + b[3] - y_test) ** 2)

configs = [
    ("No regularization",           dict()),
    ("L2 only (λ=0.01)",            dict(use_l2=True)),
    ("Dropout only (p=0.3)",        dict(use_dropout=True)),
    ("Early stopping (pat=200)",    dict(use_early_stop=True)),
    ("Label smoothing (ε=0.1)",     dict(use_label_smooth=True)),
    ("L2 + Dropout",                dict(use_l2=True, use_dropout=True)),
    ("L2 + Dropout + Early stop",   dict(use_l2=True, use_dropout=True, use_early_stop=True)),
]

print(f"{'Config':<30s} {'Test Loss':>10s}")
print("-" * 42)
for name, kwargs in configs:
    loss = train_network(**kwargs)
    print(f"{name:<30s} {loss:>10.4f}")
