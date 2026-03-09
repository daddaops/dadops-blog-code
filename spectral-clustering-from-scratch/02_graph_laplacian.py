"""Compute graph Laplacian and examine its spectrum."""
import numpy as np
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=200, noise=0.06, random_state=42)

def rbf_similarity(X, sigma=0.1):
    """Fully connected similarity matrix with RBF kernel."""
    n = X.shape[0]
    dists_sq = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    return np.exp(-dists_sq / (2 * sigma ** 2))

def compute_laplacian(W, normalized=False):
    """Compute graph Laplacian from similarity matrix W."""
    D = np.diag(W.sum(axis=1))
    L = D - W

    if not normalized:
        return L

    # Symmetric normalized Laplacian: D^(-1/2) L D^(-1/2)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))
    L_sym = d_inv_sqrt @ L @ d_inv_sqrt
    return L_sym

if __name__ == "__main__":
    # Build Laplacian and examine its spectrum
    W = rbf_similarity(X, sigma=0.1)
    L = compute_laplacian(W, normalized=False)
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    print("First 10 eigenvalues:")
    print(np.round(eigenvalues[:10], 6))

    print(f"\nEigengap (lambda_3 - lambda_2): {eigenvalues[2] - eigenvalues[1]:.4f}")
    print(f"This suggests k=2 clusters")
