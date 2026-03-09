"""Label propagation through an RBF similarity graph."""
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


def label_propagation(X_all, y_known, labeled_mask, sigma=0.3,
                      max_iter=50):
    """Propagate labels through an RBF similarity graph."""
    n = len(X_all)
    n_classes = int(y_known.max()) + 1

    # Build RBF weight matrix
    diff = X_all[:, None, :] - X_all[None, :, :]
    W = np.exp(-np.sum(diff**2, axis=2) / (2 * sigma**2))
    np.fill_diagonal(W, 0)  # no self-loops

    # Transition matrix: T = D^{-1} W
    D_inv = 1.0 / W.sum(axis=1)
    T = W * D_inv[:, None]

    # Initialize label matrix Y (one-hot for labeled, zero for unlabeled)
    Y = np.zeros((n, n_classes))
    for i in range(n):
        if labeled_mask[i]:
            Y[i, int(y_known[i])] = 1.0

    Y_init = Y.copy()

    # Iterate: propagate then clamp
    for _ in range(max_iter):
        Y = T @ Y
        Y[labeled_mask] = Y_init[labeled_mask]  # clamp labeled nodes

    return Y.argmax(axis=1), Y


if __name__ == "__main__":
    # Two-moons data: 520 points, 10 labeled per class
    X, y_true = make_moons(260, noise=0.15, seed=7)
    labeled_mask = np.zeros(520, dtype=bool)
    rng = np.random.RandomState(7)
    for cls in [0, 1]:
        idx = np.where(y_true == cls)[0]
        labeled_mask[rng.choice(idx, 10, replace=False)] = True

    y_known = np.where(labeled_mask, y_true, -1).astype(float)
    preds, soft_labels = label_propagation(X, y_known, labeled_mask,
                                            sigma=0.3)
    acc = np.mean(preds == y_true)
    print(f"Label propagation accuracy: {acc:.1%}")

    # How sigma affects accuracy
    for s in [0.05, 0.1, 0.3, 0.5, 1.0, 3.0]:
        p, _ = label_propagation(X, y_known, labeled_mask, sigma=s)
        print(f"  sigma={s:.2f}: {np.mean(p == y_true):.1%}")
