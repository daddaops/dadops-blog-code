"""
The Interpolation Peak and Regularization

Sweeps the parameter/data ratio from 0.1 to 5.0 and shows how ridge
regularization (lambda > 0) smooths the interpolation peak at p/n ≈ 1.
"""
import numpy as np

np.random.seed(42)
n = 30
x = np.sort(np.random.uniform(0, 2 * np.pi, n))
y = np.sin(x) + np.random.randn(n) * 0.4
x_test = np.linspace(0, 2 * np.pi, 300)
y_test = np.sin(x_test)

ratios = np.concatenate([np.linspace(0.1, 0.9, 9), np.linspace(0.95, 1.05, 11), np.linspace(1.1, 5.0, 20)])
results = {lam: [] for lam in [0, 0.01, 0.1]}

for ratio in ratios:
    p = max(2, int(ratio * n))
    X_tr = np.vander(x, p + 1, increasing=True)
    X_te = np.vander(x_test, p + 1, increasing=True)

    for lam in [0, 0.01, 0.1]:
        if p <= n:
            w = np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(p + 1), X_tr.T @ y)
        else:
            w = X_tr.T @ np.linalg.solve(X_tr @ X_tr.T + lam * np.eye(n), y)
        test_mse = np.mean((X_te @ w - y_test) ** 2)
        results[lam].append((ratio, min(test_mse, 5.0)))  # cap for display

# Print key values showing the peak and its smoothing
for lam in [0, 0.01, 0.1]:
    vals = results[lam]
    peak = max(vals, key=lambda v: v[1])
    final = vals[-1]
    print(f"lambda={lam:.2f}  peak_error={peak[1]:.3f} at p/n={peak[0]:.2f}  "
          f"final_error={final[1]:.3f} at p/n={final[0]:.2f}")


if __name__ == "__main__":
    pass  # Output printed above
