"""
Minimum-Norm Interpolation

Compares unregularized minimum-norm solution to optimally-tuned ridge
regression in the overparameterized regime (p = 5n).
"""
import numpy as np

np.random.seed(42)
n, p = 30, 150  # 5x overparameterized
X = np.random.randn(n, p)
true_w = np.zeros(p)
true_w[:5] = [1, -0.5, 0.3, -0.8, 0.6]  # only 5 features matter
y = X @ true_w + np.random.randn(n) * 0.3

X_test = np.random.randn(500, p)
y_test = X_test @ true_w

# Minimum-norm (no regularization): w = X^T (X X^T)^{-1} y
w_mn = X.T @ np.linalg.solve(X @ X.T + 1e-12 * np.eye(n), y)

# Ridge regression with optimal lambda (found by cross-validation)
best_lam, best_err = 0, float('inf')
for lam in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
    w_ridge = X.T @ np.linalg.solve(X @ X.T + lam * np.eye(n), y)
    err = np.mean((X_test @ w_ridge - y_test) ** 2)
    if err < best_err:
        best_lam, best_err = lam, err

w_ridge = X.T @ np.linalg.solve(X @ X.T + best_lam * np.eye(n), y)

print(f"Minimum-norm:  test_MSE={np.mean((X_test @ w_mn - y_test)**2):.4f}  ||w||={np.linalg.norm(w_mn):.3f}")
print(f"Ridge (lam={best_lam}): test_MSE={np.mean((X_test @ w_ridge - y_test)**2):.4f}  ||w||={np.linalg.norm(w_ridge):.3f}")
print(f"Train error:   min-norm={np.mean((X @ w_mn - y)**2):.6f}  ridge={np.mean((X @ w_ridge - y)**2):.6f}")


if __name__ == "__main__":
    pass  # Output printed above
