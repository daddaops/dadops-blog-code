"""
Filter Methods: Variance, Correlation, and Mutual Information

Demonstrates three filter-based feature selection methods that rank
features independently of any model.

Blog post: https://dadops.dev/blog/feature-selection-from-scratch/
"""
import numpy as np
from dataset import make_dataset

X, y, names, n = make_dataset()

# --- Variance threshold ---
variances = X.var(axis=0)
print("Variances:", [f"{names[i]}={variances[i]:.3f}" for i in range(10)])

# --- Pearson correlation with target ---
cors = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(10)])
print("\n|Correlation| ranking:")
for idx in np.argsort(-np.abs(cors)):
    print(f"  {names[idx]:<10s} |r| = {abs(cors[idx]):.3f}")

# --- Mutual information (binned estimator) ---
def estimate_mi(x, y, bins=10):
    """MI between continuous x and binary y via histogram."""
    c, edges = np.histogram(x, bins=bins)
    mi = 0.0
    for label in [0, 1]:
        mask = y == label
        p_y = mask.mean()
        if p_y == 0:
            continue
        c_xy, _ = np.histogram(x[mask], bins=edges)
        for b in range(bins):
            p_x = c[b] / n
            p_xy = c_xy[b] / n
            if p_xy > 0 and p_x > 0:
                mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi

mis = np.array([estimate_mi(X[:, j], y) for j in range(10)])
print("\nMutual Information ranking:")
for idx in np.argsort(-mis):
    print(f"  {names[idx]:<10s} MI = {mis[idx]:.3f} bits")
