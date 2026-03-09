"""PCA via SVD — numerically stable, no covariance matrix needed.

Verifies SVD and eigendecomposition produce identical results.
"""
import numpy as np

def pca_from_scratch(X, n_components):
    """PCA via eigendecomposition."""
    n_samples = X.shape[0]
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov_matrix = (X_centered.T @ X_centered) / n_samples
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :n_components].T
    X_projected = X_centered @ components.T
    total_var = eigenvalues.sum()
    explained_var_ratio = eigenvalues[:n_components] / total_var
    return X_projected, components, explained_var_ratio

def pca_via_svd(X, n_components):
    """PCA using SVD — numerically stable, no covariance matrix needed."""
    n_samples = X.shape[0]

    # Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # SVD of centered data
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # V's rows (Vt) are the principal components
    components = Vt[:n_components]

    # Project data
    X_projected = X_centered @ components.T

    # Explained variance from singular values
    explained_var = (S ** 2) / n_samples
    total_var = explained_var.sum()
    explained_var_ratio = explained_var[:n_components] / total_var

    return X_projected, components, explained_var_ratio


# Create the same 2D data
np.random.seed(42)
n_points = 200
angle = np.pi / 4
stretch = np.array([[3, 0], [0, 0.5]])
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])
transform = rotation @ stretch
raw = np.random.randn(n_points, 2)
X = raw @ transform.T + np.array([5, 5])

_, components, _ = pca_from_scratch(X, n_components=2)

# Verify: both methods give the same result
X_proj_svd, comp_svd, var_svd = pca_via_svd(X, n_components=2)
print("SVD components match eigendecomposition:",
      np.allclose(np.abs(components), np.abs(comp_svd)))
print(f"SVD variance ratios: {var_svd}")
# Note: eigenvectors may differ in sign (both v and -v are valid eigenvectors)
