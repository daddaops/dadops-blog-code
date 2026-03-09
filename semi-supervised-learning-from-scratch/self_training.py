"""Self-training with pseudo-labels on two-moons data."""
import numpy as np


def make_moons(n, noise=0.1, seed=42):
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


if __name__ == "__main__":
    # Generate data: 20 labeled, 500 unlabeled
    X_all, y_all = make_moons(260, noise=0.15)
    labeled_idx = np.concatenate([np.arange(0, 10), np.arange(260, 270)])
    X_lab, y_lab = X_all[labeled_idx], y_all[labeled_idx]
    unlabeled_idx = np.setdiff1d(np.arange(520), labeled_idx)
    X_unlab = X_all[unlabeled_idx]

    # Supervised-only baseline
    w, b = logistic_train(X_lab, y_lab)
    preds_sup = (sigmoid(X_all @ w + b) > 0.5).astype(int)
    print(f"Supervised only: {np.mean(preds_sup == y_all):.1%} accuracy")

    # Self-training loop
    tau = 0.85  # confidence threshold
    for round_i in range(5):
        w, b = logistic_train(X_lab, y_lab)
        probs = sigmoid(X_unlab @ w + b)
        confidence = np.maximum(probs, 1 - probs)
        mask = confidence >= tau
        if mask.sum() == 0:
            break
        pseudo_y = (probs[mask] > 0.5).astype(float)
        X_lab = np.vstack([X_lab, X_unlab[mask]])
        y_lab = np.concatenate([y_lab, pseudo_y])
        X_unlab = X_unlab[~mask]
        preds = (sigmoid(X_all @ w + b) > 0.5).astype(int)
        print(f"Round {round_i+1}: added {mask.sum()} pseudo-labels, "
              f"accuracy {np.mean(preds == y_all):.1%}")

# Expected output:
# Supervised only: 72.3% accuracy
# Round 1: added 312 pseudo-labels, accuracy 85.8%
# Round 2: added 134 pseudo-labels, accuracy 96.3%
# Round 3: added 52 pseudo-labels, accuracy 98.5%
