"""
Stability Selection

Bootstrap resampling with Lasso to identify features that are
consistently selected across subsamples.

Blog post: https://dadops.dev/blog/feature-selection-from-scratch/
"""
import numpy as np
from dataset import make_dataset
from lasso_path import lasso_coordinate_descent

X, y, names, n = make_dataset()


def stability_selection(X, y, n_bootstrap=50, lam=0.05, threshold=0.6):
    """Stability selection: bootstrap Lasso, count selection frequencies."""
    n, d = X.shape
    selection_counts = np.zeros(d)
    rng = np.random.RandomState(42)

    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_c = y - y.mean()

    for b in range(n_bootstrap):
        # Subsample 50% of data
        idx = rng.choice(n, size=n // 2, replace=False)
        w = lasso_coordinate_descent(X_std[idx], y_c[idx], lam, epochs=300)
        selection_counts += (np.abs(w) > 1e-8).astype(float)

    frequencies = selection_counts / n_bootstrap
    stable_set = [j for j in range(d) if frequencies[j] >= threshold]
    return frequencies, stable_set


freqs, stable = stability_selection(X, y)

print("Selection frequencies (50 bootstrap runs):")
for idx in np.argsort(-freqs):
    tag = " STABLE" if freqs[idx] >= 0.6 else ""
    print(f"  {names[idx]:<10s} freq = {freqs[idx]:.2f}{tag}")
print(f"\nStable feature set: {[names[i] for i in stable]}")
