"""
Visualizing the Three Regimes

Uses random Fourier features to fit models with p=5 (underfit),
p=n (critical), and p=5n (overparameterized) features, showing
the three regimes of double descent.
"""
import numpy as np

np.random.seed(42)
n = 20
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + np.random.randn(n) * 0.3
x_dense = np.linspace(0, 2 * np.pi, 500)

# Fixed random frequencies for feature construction
rng = np.random.RandomState(99)
max_p = 5 * n
omegas = rng.randn(max_p)
biases = rng.uniform(0, 2 * np.pi, max_p)


def make_features(x, p):
    return np.cos(np.outer(x, omegas[:p]) + biases[:p])


for label, p in [("UNDERFIT (p=5)", 5), ("CRITICAL (p=n)", n), ("OVERFIT-FREE (p=5n)", 5*n)]:
    X = make_features(x, p)
    X_dense = make_features(x_dense, p)

    if p < n:
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    elif p == n:
        # At interpolation threshold: exact solve
        w = np.linalg.solve(X, y)
    else:
        # Minimum-norm solution
        w = X.T @ np.linalg.solve(X @ X.T + 1e-12 * np.eye(n), y)

    y_pred = X_dense @ w
    y_pred_clipped = np.clip(y_pred, -3, 3)
    test_err = np.mean((y_pred_clipped - np.sin(x_dense)) ** 2)
    train_err = np.mean((X @ w - y) ** 2)
    print(f"{label:>25s}  train={train_err:.6f}  test={test_err:.4f}  ||w||={np.linalg.norm(w):.3f}")


if __name__ == "__main__":
    pass  # Output printed above
