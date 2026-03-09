"""Entropy minimization for semi-supervised learning."""
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


def logistic_train(X, y, lr=0.5, steps=200):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        grad_w = X.T @ (p - y) / len(y)
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def entropy_loss(probs):
    """Binary entropy: -p*log(p) - (1-p)*log(1-p)."""
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def train_with_entropy_min(X_lab, y_lab, X_unlab, lam=1.0,
                           lr=0.5, steps=300):
    """Logistic regression with entropy minimization on unlabeled data."""
    X_all = np.vstack([X_lab, X_unlab])
    n_lab = len(y_lab)
    w = np.zeros(X_lab.shape[1])
    b = 0.0

    for _ in range(steps):
        p_all = sigmoid(X_all @ w + b)
        p_lab = p_all[:n_lab]
        p_unlab = p_all[n_lab:]

        # Supervised gradient: d/dw of cross-entropy on labeled
        grad_sup_w = X_lab.T @ (p_lab - y_lab) / n_lab
        grad_sup_b = np.mean(p_lab - y_lab)

        # Entropy gradient: d/dz of H(sigma(z))
        # = (log(1-p) - log(p)) * p * (1-p)
        # Pushes predictions away from 0.5 (toward confident 0 or 1)
        p_clip = np.clip(p_unlab, 1e-7, 1 - 1e-7)
        ent_grad = (np.log(1 - p_clip) - np.log(p_clip)) * p_clip * (1 - p_clip)
        grad_ent_w = lam * X_unlab.T @ ent_grad / len(p_unlab)
        grad_ent_b = lam * np.mean(ent_grad)

        w -= lr * (grad_sup_w + grad_ent_w)
        b -= lr * (grad_sup_b + grad_ent_b)

    return w, b


if __name__ == "__main__":
    # Compare with and without entropy minimization
    X, y = make_moons(260, noise=0.15, seed=7)
    rng = np.random.RandomState(7)
    idx_l = np.concatenate([
        rng.choice(np.where(y == 0)[0], 5, replace=False),
        rng.choice(np.where(y == 1)[0], 5, replace=False)
    ])
    X_l, y_l = X[idx_l], y[idx_l]
    X_u = np.delete(X, idx_l, axis=0)

    w0, b0 = logistic_train(X_l, y_l)
    acc0 = np.mean((sigmoid(X @ w0 + b0) > 0.5) == y)

    w1, b1 = train_with_entropy_min(X_l, y_l, X_u, lam=0.5)
    acc1 = np.mean((sigmoid(X @ w1 + b1) > 0.5) == y)

    print(f"Supervised (10 labels):  {acc0:.1%}")
    print(f"+ Entropy min:           {acc1:.1%}")
    print(f"Mean entropy (sup only): {entropy_loss(sigmoid(X @ w0 + b0)).mean():.3f}")
    print(f"Mean entropy (+ ent min):{entropy_loss(sigmoid(X @ w1 + b1)).mean():.3f}")
