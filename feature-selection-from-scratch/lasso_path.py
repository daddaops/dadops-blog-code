"""
Lasso Feature Selection via Coordinate Descent

Implements Lasso regression with soft thresholding and sweeps the
regularization parameter to show the feature selection path.

Blog post: https://dadops.dev/blog/feature-selection-from-scratch/
"""
import numpy as np
from dataset import make_dataset

def lasso_coordinate_descent(X, y, lam, epochs=500):
    """Lasso via coordinate descent. Returns coefficients."""
    n, d = X.shape
    w = np.zeros(d)
    # Precompute for efficiency
    X_col_sq = (X ** 2).sum(axis=0)
    for _ in range(epochs):
        for j in range(d):
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual
            # Soft thresholding
            if rho > lam * n:
                w[j] = (rho - lam * n) / X_col_sq[j]
            elif rho < -lam * n:
                w[j] = (rho + lam * n) / X_col_sq[j]
            else:
                w[j] = 0.0
    return w


if __name__ == "__main__":
    X, y, names, n = make_dataset()

    # Standardize features for fair penalization
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_centered = y - y.mean()

    # Regularization path: sweep lambda from high to low
    lambdas = np.logspace(0, -2, 30)
    paths = np.zeros((30, 10))

    for i, lam in enumerate(lambdas):
        paths[i] = lasso_coordinate_descent(X_std, y_centered, lam)

    print("Lambda    Nonzero features (selected)")
    print("-" * 50)
    for i in [0, 5, 10, 15, 20, 29]:
        nonzero = [names[j] for j in range(10) if abs(paths[i, j]) > 1e-8]
        print(f"  {lambdas[i]:.4f}   [{len(nonzero)}] {nonzero}")
