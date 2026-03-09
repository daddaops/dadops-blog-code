"""
Double Descent Curve with Polynomial Regression

Fits polynomials of degree 1 to ~120 on a noisy sine wave (n=30 data points).
Shows the three regimes: underfitting, interpolation peak, overparameterized descent.
"""
import numpy as np

np.random.seed(42)
n = 30
x_train = np.linspace(0, 2 * np.pi, n)
y_train = np.sin(x_train) + np.random.randn(n) * 0.3
x_test = np.linspace(0, 2 * np.pi, 200)
y_test = np.sin(x_test)

degrees = list(range(1, 25)) + list(range(25, 120, 3))
train_errors, test_errors = [], []

for d in degrees:
    # Vandermonde matrix: each column is x^k
    X_tr = np.vander(x_train, d + 1, increasing=True)
    X_te = np.vander(x_test, d + 1, increasing=True)

    if d < n:
        # Overdetermined: least-squares solution
        w, _, _, _ = np.linalg.lstsq(X_tr, y_train, rcond=None)
    else:
        # Underdetermined: minimum-norm solution
        # w = X^T (X X^T)^{-1} y
        w = X_tr.T @ np.linalg.solve(X_tr @ X_tr.T + 1e-12 * np.eye(n), y_train)

    train_errors.append(np.mean((X_tr @ w - y_train) ** 2))
    test_errors.append(np.mean((X_te @ w - y_test) ** 2))

# Plot degrees vs test_errors to see the double descent curve
# Peak occurs near degree = n (interpolation threshold)
for d, tr, te in zip(degrees, train_errors, test_errors):
    regime = "UNDER" if d < 15 else ("CRITICAL" if 25 <= d <= 35 else "OVER")
    if d in [3, 10, 28, 30, 50, 100]:
        print(f"degree={d:>3d}  train={tr:.4f}  test={te:.4f}  [{regime}]")


if __name__ == "__main__":
    pass  # Output printed above
