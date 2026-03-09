"""PCA from scratch using eigendecomposition of the covariance matrix.

Demonstrates the 5-step PCA algorithm on a 2D elliptical dataset.
"""
import numpy as np

def pca_from_scratch(X, n_components):
    """
    Principal Component Analysis from scratch.

    Parameters:
        X: ndarray of shape (n_samples, n_features)
        n_components: number of principal components to keep

    Returns:
        X_projected: data in the reduced space (n_samples, n_components)
        components: the principal component directions (n_components, n_features)
        explained_var_ratio: fraction of variance each component explains
    """
    n_samples = X.shape[0]

    # Step 1: center the data (subtract feature means)
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2: compute the covariance matrix
    # Using (1/n) instead of (1/(n-1)) for population covariance
    cov_matrix = (X_centered.T @ X_centered) / n_samples

    # Step 3: eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # eigh returns eigenvalues in ascending order — flip to descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: select the top-k eigenvectors (principal components)
    components = eigenvectors[:, :n_components].T  # shape: (n_components, n_features)

    # Step 5: project data onto the new subspace
    X_projected = X_centered @ components.T

    # Compute explained variance ratios
    total_var = eigenvalues.sum()
    explained_var_ratio = eigenvalues[:n_components] / total_var

    return X_projected, components, explained_var_ratio


# Demo: 2D elliptical data
np.random.seed(42)
n_points = 200

# Create correlated 2D data (tilted ellipse)
angle = np.pi / 4  # 45-degree tilt
stretch = np.array([[3, 0], [0, 0.5]])  # long along x, short along y
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])
transform = rotation @ stretch

raw = np.random.randn(n_points, 2)
X = raw @ transform.T + np.array([5, 5])  # shift away from origin

X_proj, components, var_ratios = pca_from_scratch(X, n_components=2)

print("Principal components (directions):")
print(components)
print(f"\nExplained variance ratios: {var_ratios}")
print(f"PC1 captures {var_ratios[0]:.1%} of the variance")
print(f"PC2 captures {var_ratios[1]:.1%} of the variance")
