"""
The Interpolation Peak and Regularization

Sweeps the parameter/data ratio (p/n) using iid Gaussian random features
and shows how ridge regularization (lambda > 0) smooths the interpolation
peak at p/n ~ 1. Uses the standard setup from Hastie et al. (2022).
"""
import numpy as np

n = 80  # enough data for smooth curves
sigma = 0.3  # noise level
num_trials = 10  # average over trials for smoother curves

ratios = np.concatenate([
    np.linspace(0.1, 0.85, 8),
    np.linspace(0.9, 1.15, 26),  # dense near p/n = 1
    np.linspace(1.2, 4.0, 12)
])
results = {lam: [] for lam in [0, 0.01, 0.1]}

for ratio in ratios:
    p = max(2, int(ratio * n))
    trial_results = {lam: [] for lam in [0, 0.01, 0.1]}

    for trial in range(num_trials):
        rng = np.random.RandomState(1000 * trial + int(ratio * 100))
        # iid Gaussian design (the standard theoretical setting)
        X_tr = rng.randn(n, p) / np.sqrt(n)
        X_te = rng.randn(200, p) / np.sqrt(n)
        # Sparse true signal
        w_true = np.zeros(p)
        w_true[:min(p, 5)] = np.array([1, -0.5, 0.3, -0.8, 0.6])[:min(p, 5)]
        y_tr = X_tr @ w_true + rng.randn(n) * sigma
        y_te = X_te @ w_true

        for lam in [0, 0.01, 0.1]:
            if p <= n:
                A = X_tr.T @ X_tr + max(lam, 1e-12) * np.eye(p)
                w = np.linalg.solve(A, X_tr.T @ y_tr)
            else:
                A = X_tr @ X_tr.T + max(lam, 1e-12) * np.eye(n)
                w = X_tr.T @ np.linalg.solve(A, y_tr)
            test_mse = np.mean((X_te @ w - y_te) ** 2)
            trial_results[lam].append(test_mse)

    for lam in [0, 0.01, 0.1]:
        avg_mse = np.mean(trial_results[lam])
        results[lam].append((ratio, avg_mse))

# Print key values showing the peak and its smoothing
print("Interpolation Peak and Regularization (n=80, averaged over 10 trials)")
print("=" * 72)
for lam in [0, 0.01, 0.1]:
    vals = results[lam]
    peak = max(vals, key=lambda v: v[1])
    final = vals[-1]
    over_vals = [(r, e) for r, e in vals if r > 1.2]
    best_over = min(over_vals, key=lambda v: v[1]) if over_vals else final
    under_vals = [(r, e) for r, e in vals if r < 0.85]
    best_under = min(under_vals, key=lambda v: v[1]) if under_vals else vals[0]
    print(f"lambda={lam:.2f}  peak={peak[1]:.4f} at p/n={peak[0]:.2f}  "
          f"best_under={best_under[1]:.4f}  best_over={best_over[1]:.4f}  "
          f"final={final[1]:.4f}")

# Detailed view near the peak
print(f"\nDetailed near p/n=1 (lambda=0):")
for ratio, mse in results[0]:
    if 0.85 <= ratio <= 1.2:
        print(f"  p/n={ratio:.3f}  test_MSE={mse:.4f}")


if __name__ == "__main__":
    pass  # Output printed above
