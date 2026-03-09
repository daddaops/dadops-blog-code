"""Shared helpers for pruning scripts."""
import numpy as np

def make_spiral_data(n_per_class=150, n_classes=3, seed=0):
    """Generate a 3-class spiral dataset for classification."""
    np.random.seed(seed)
    X, y = [], []
    for k in range(n_classes):
        r = np.linspace(0.2, 1.0, n_per_class)
        theta = np.linspace(k * 4.2, (k + 1) * 4.2, n_per_class) + np.random.randn(n_per_class) * 0.25
        X.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        y.append(np.full(n_per_class, k, dtype=int))
    return np.vstack(X), np.concatenate(y)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def train_mlp(X, y, hidden=[64, 32], lr=0.05, epochs=400):
    """Train a simple MLP: 2 -> 64 -> 32 -> 3."""
    np.random.seed(42)
    n_classes = y.max() + 1
    dims = [X.shape[1]] + hidden + [n_classes]
    W, b = [], []
    for i in range(len(dims) - 1):
        W.append(np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i]))
        b.append(np.zeros(dims[i+1]))
    for epoch in range(epochs):
        h = X
        activations = [h]
        for i in range(len(W) - 1):
            h = np.maximum(0, h @ W[i] + b[i])
            activations.append(h)
        logits = h @ W[-1] + b[-1]
        probs = softmax(logits)
        dz = probs.copy()
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y)), y] = 1
        dz -= one_hot
        dz /= len(y)
        for i in range(len(W) - 1, -1, -1):
            dW = activations[i].T @ dz
            db = dz.sum(axis=0)
            if i > 0:
                dz = (dz @ W[i].T) * (activations[i] > 0)
            W[i] -= lr * dW
            b[i] -= lr * db
    return W, b

def evaluate(W, b, X, y):
    """Forward pass and compute accuracy."""
    h = X
    for i in range(len(W) - 1):
        h = np.maximum(0, h @ W[i] + b[i])
    logits = h @ W[-1] + b[-1]
    return (logits.argmax(axis=1) == y).mean()

def magnitude_prune(W, sparsity):
    """Global magnitude pruning: zero out the smallest weights across all layers."""
    all_weights = np.concatenate([w.flatten() for w in W])
    threshold = np.percentile(np.abs(all_weights), sparsity * 100)
    return [np.where(np.abs(w) >= threshold, w, 0.0) for w in W]
