"""
Visualizing the Three Regimes

Fits polynomials of degree 5, n, and 5n to show the underfit, critical,
and overparameterized regimes visually.
"""
import numpy as np

np.random.seed(42)
n = 20
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + np.random.randn(n) * 0.3
x_dense = np.linspace(0, 2 * np.pi, 500)

for label, d in [("UNDERFIT (p=5)", 5), ("CRITICAL (p=n)", n), ("OVERFIT-FREE (p=5n)", 5*n)]:
    X = np.vander(x, d + 1, increasing=True)
    X_dense = np.vander(x_dense, d + 1, increasing=True)

    if d < n:
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    else:
        w = X.T @ np.linalg.solve(X @ X.T + 1e-12 * np.eye(n), y)

    y_pred = X_dense @ w
    # Clip extreme values for display
    y_pred = np.clip(y_pred, -3, 3)
    test_err = np.mean((y_pred - np.sin(x_dense)) ** 2)
    train_err = np.mean((X @ w - y) ** 2)
    print(f"{label:>25s}  train={train_err:.4f}  test={test_err:.4f}  ||w||={np.linalg.norm(w):.2f}")


if __name__ == "__main__":
    pass  # Output printed above
