"""Simplified MixMatch: augment-average-sharpen-mixup."""
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


def mixmatch_train(X_lab, y_lab, X_unlab, K=2, T=0.5,
                   alpha=0.75, lr=0.3, epochs=400, lam_u=1.0):
    """Simplified MixMatch: augment-average-sharpen-mixup."""
    rng = np.random.RandomState(42)
    n_lab, n_feat = X_lab.shape
    w = rng.randn(n_feat) * 0.1
    b = 0.0

    def predict(X):
        return sigmoid(X @ w + b)

    def augment(X):
        return X + rng.randn(*X.shape) * 0.15

    for _ in range(epochs):
        # Step 1: Augment-average predictions for unlabeled
        avg_pred = np.zeros(len(X_unlab))
        for _k in range(K):
            avg_pred += predict(augment(X_unlab))
        avg_pred /= K

        # Step 2: Sharpen (temperature scaling for binary case)
        sharp = avg_pred ** (1/T)
        q = sharp / (sharp + (1 - avg_pred) ** (1/T))

        # Step 3: MixUp — combine labeled and pseudo-labeled
        X_combined = np.vstack([X_lab, X_unlab])
        y_combined = np.concatenate([y_lab, q])
        perm = rng.permutation(len(X_combined))
        lam = rng.beta(alpha, alpha)
        lam_p = max(lam, 1 - lam)  # keep closer to original

        X_mix = lam_p * X_combined + (1 - lam_p) * X_combined[perm]
        y_mix = lam_p * y_combined + (1 - lam_p) * y_combined[perm]

        # Step 4: Train — supervised CE + unsupervised L2
        p_mix = predict(X_mix)
        n_total = len(X_mix)

        # Gradient: supervised part (first n_lab mixed examples)
        err_s = p_mix[:n_lab] - y_mix[:n_lab]
        # Gradient: unsupervised part (L2 loss, bounded gradient)
        err_u = lam_u * 2 * (p_mix[n_lab:] - y_mix[n_lab:])

        err = np.concatenate([err_s, err_u]) * p_mix * (1 - p_mix)
        w -= lr * X_mix.T @ err / n_total
        b -= lr * np.mean(err)

    return w, b


if __name__ == "__main__":
    # Full comparison on two-moons
    X, y = make_moons(260, noise=0.15, seed=42)
    rng = np.random.RandomState(42)
    idx = np.concatenate([
        rng.choice(np.where(y == 0)[0], 10, replace=False),
        rng.choice(np.where(y == 1)[0], 10, replace=False)
    ])
    X_l, y_l = X[idx], y[idx]
    X_u = np.delete(X, idx, axis=0)

    w_sup, b_sup = logistic_train(X_l, y_l)
    w_mm, b_mm = mixmatch_train(X_l, y_l, X_u)

    acc_sup = np.mean((sigmoid(X @ w_sup + b_sup) > 0.5) == y)
    acc_mm = np.mean((sigmoid(X @ w_mm + b_mm) > 0.5) == y)
    print(f"Supervised (20 labels):  {acc_sup:.1%}")
    print(f"MixMatch   (20 labels):  {acc_mm:.1%}")
