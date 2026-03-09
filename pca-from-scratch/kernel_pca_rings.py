"""Kernel PCA for nonlinear structure (concentric rings).

Standard PCA can't separate concentric rings; Kernel PCA with
RBF kernel maps them to a higher-dimensional space where they're linear.
"""
import numpy as np

def pca_via_svd(X, n_components):
    """PCA using SVD."""
    n_samples = X.shape[0]
    mean = X.mean(axis=0)
    X_centered = X - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    X_projected = X_centered @ components.T
    explained_var = (S ** 2) / n_samples
    total_var = explained_var.sum()
    explained_var_ratio = explained_var[:n_components] / total_var
    return X_projected, components, explained_var_ratio

def make_concentric_rings(n_per_ring=200, noise=0.1, seed=42):
    """Generate two concentric rings in 2D — a nonlinear structure."""
    rng = np.random.RandomState(seed)
    angles_inner = rng.uniform(0, 2 * np.pi, n_per_ring)
    angles_outer = rng.uniform(0, 2 * np.pi, n_per_ring)

    r_inner, r_outer = 1.0, 3.0
    inner = np.column_stack([
        r_inner * np.cos(angles_inner) + rng.randn(n_per_ring) * noise,
        r_inner * np.sin(angles_inner) + rng.randn(n_per_ring) * noise
    ])
    outer = np.column_stack([
        r_outer * np.cos(angles_outer) + rng.randn(n_per_ring) * noise,
        r_outer * np.sin(angles_outer) + rng.randn(n_per_ring) * noise
    ])

    X = np.vstack([inner, outer])
    labels = np.array([0] * n_per_ring + [1] * n_per_ring)
    return X, labels


def kernel_pca(X, n_components=2, kernel='rbf', gamma=1.0):
    """Kernel PCA using the RBF (Gaussian) kernel."""
    n = X.shape[0]

    # Compute kernel matrix
    if kernel == 'rbf':
        sq_dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
        K = np.exp(-gamma * sq_dists)
    else:
        K = X @ X.T  # linear kernel fallback

    # Center the kernel matrix
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Normalize eigenvectors by sqrt of eigenvalues
    X_kpca = eigenvectors[:, :n_components]
    for i in range(n_components):
        if eigenvalues[i] > 0:
            X_kpca[:, i] *= np.sqrt(eigenvalues[i])

    return X_kpca


X_rings, labels = make_concentric_rings()

# Standard PCA
X_pca, _, _ = pca_via_svd(X_rings, n_components=2)

# Kernel PCA with RBF kernel
X_kpca = kernel_pca(X_rings, n_components=2, gamma=0.5)

print("Concentric rings: PCA sees overlapping projections along both axes.")
print("Kernel PCA separates them by mapping to a higher-dimensional space.")
print(f"Data shape: {X_rings.shape}, Labels: {np.unique(labels)}")
print(f"PCA projection range: [{X_pca[:, 0].min():.2f}, {X_pca[:, 0].max():.2f}]")
print(f"Kernel PCA PC1 range: [{X_kpca[:, 0].min():.2f}, {X_kpca[:, 0].max():.2f}]")
