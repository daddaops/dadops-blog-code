"""Pi-Model consistency regularization for semi-supervised learning."""
import numpy as np


def make_moons(n, noise=0.1, seed=42):
    """Generate two interleaving half-moons."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, np.pi, n)
    x1 = np.c_[np.cos(t), np.sin(t)] + rng.randn(n, 2) * noise
    x2 = np.c_[1 - np.cos(t), -np.sin(t) + 0.5] + rng.randn(n, 2) * noise
    X = np.vstack([x1, x2])
    y = np.array([0]*n + [1]*n)
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def pi_model_train(X_lab, y_lab, X_unlab, hidden=32, lr=0.01,
                   epochs=300, noise_std=0.3, w_max=1.0):
    """Train a simple network with Pi-Model consistency."""
    rng = np.random.RandomState(42)
    n_feat = X_lab.shape[1]

    # Initialize weights (single hidden layer)
    W1 = rng.randn(n_feat, hidden) * 0.5
    b1 = np.zeros(hidden)
    W2 = rng.randn(hidden, 1) * 0.5
    b2 = np.zeros(1)
    X_all = np.vstack([X_lab, X_unlab])

    for epoch in range(epochs):
        # Ramp-up weight: Gaussian schedule
        t = epoch / epochs
        w_cons = w_max * np.exp(-5 * (1 - t)**2) if epoch > 10 else 0

        # Forward pass 1: with noise
        noise1 = rng.randn(*X_all.shape) * noise_std
        h1 = np.maximum(0, (X_all + noise1) @ W1 + b1)  # ReLU
        p1 = sigmoid(h1 @ W2 + b2).ravel()

        # Forward pass 2: with different noise
        noise2 = rng.randn(*X_all.shape) * noise_std
        h2 = np.maximum(0, (X_all + noise2) @ W1 + b1)
        p2 = sigmoid(h2 @ W2 + b2).ravel()

        # Supervised loss gradient (labeled only)
        n_lab = len(y_lab)
        err_lab = p1[:n_lab] - y_lab

        # Consistency loss gradient (all data): d/dp1 of (p1 - p2)^2
        err_cons = 2 * (p1 - p2) * w_cons

        err_full = np.zeros(len(X_all))
        err_full[:n_lab] = err_lab
        err_full += err_cons / len(X_all) * n_lab  # scale consistently

        # Backprop through pass 1
        delta2 = err_full[:, None] * p1[:, None] * (1 - p1[:, None])
        grad_W2 = h1.T @ delta2 / n_lab
        grad_b2 = delta2.mean(axis=0)
        delta1 = (delta2 @ W2.T) * (h1 > 0)
        grad_W1 = (X_all + noise1).T @ delta1 / n_lab
        grad_b1 = delta1.mean(axis=0)

        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1

    return W1, b1, W2, b2


if __name__ == "__main__":
    # Compare: supervised only vs Pi-Model
    X, y = make_moons(260, noise=0.15, seed=99)
    rng = np.random.RandomState(99)
    idx_lab = np.concatenate([
        rng.choice(np.where(y == 0)[0], 10, replace=False),
        rng.choice(np.where(y == 1)[0], 10, replace=False)
    ])
    X_lab, y_lab = X[idx_lab], y[idx_lab]
    X_unlab = np.delete(X, idx_lab, axis=0)

    # Supervised only (w_max=0 disables consistency)
    W1, b1, W2, b2 = pi_model_train(X_lab, y_lab, X_unlab, w_max=0.0)
    p_sup = sigmoid(np.maximum(0, X @ W1 + b1) @ W2 + b2).ravel()
    print(f"Supervised only:  {np.mean((p_sup > 0.5) == y):.1%}")

    # With consistency regularization
    W1, b1, W2, b2 = pi_model_train(X_lab, y_lab, X_unlab, w_max=1.0)
    p_ssl = sigmoid(np.maximum(0, X @ W1 + b1) @ W2 + b2).ravel()
    print(f"Pi-Model (SSL):   {np.mean((p_ssl > 0.5) == y):.1%}")
