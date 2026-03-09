"""Scree plot and cumulative variance analysis.

Shows how many components are needed to capture 80/90/95/99% of variance
in 50-dimensional data with exponentially decaying variance.
"""
import numpy as np

def pca_analysis(X):
    """Compute all principal components and their explained variance."""
    n_samples = X.shape[0]
    X_centered = X - X.mean(axis=0)

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained_var = (S ** 2) / n_samples
    total_var = explained_var.sum()
    explained_var_ratio = explained_var / total_var
    cumulative_var = np.cumsum(explained_var_ratio)

    return explained_var_ratio, cumulative_var


# Example: 50-dimensional data where most variance is in few directions
np.random.seed(42)
n_samples = 500

# Create data with exponentially decaying variance across dimensions
true_dims = 50
variances = np.exp(-0.3 * np.arange(true_dims))
X_high = np.random.randn(n_samples, true_dims) * np.sqrt(variances)

var_ratios, cumulative = pca_analysis(X_high)

print("Top 10 explained variance ratios:")
for i in range(10):
    bar = "#" * int(var_ratios[i] * 200)
    print(f"  PC{i+1:2d}: {var_ratios[i]:.4f}  {bar}")

print(f"\nCumulative variance:")
for threshold in [0.80, 0.90, 0.95, 0.99]:
    n_needed = np.searchsorted(cumulative, threshold) + 1
    print(f"  {threshold:.0%} variance captured by {n_needed} components (of {true_dims})")
