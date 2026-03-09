"""Full spectral clustering pipeline compared with k-means."""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

X, y_true = make_moons(n_samples=200, noise=0.06, random_state=42)

def rbf_similarity(X, sigma=0.3):
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
    d_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))
    L_sym = d_inv_sqrt @ L @ d_inv_sqrt
    return L_sym

def spectral_clustering(X, k=2, sigma=0.3, normalized=True):
    """Full spectral clustering from scratch."""
    # Step 1: Similarity graph
    W = rbf_similarity(X, sigma)

    # Step 2: Graph Laplacian
    L = compute_laplacian(W, normalized=normalized)

    # Step 3: Eigendecomposition — k smallest eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    U = eigenvectors[:, :k]  # first k columns (smallest eigenvalues)

    # Step 4 (NJW): Normalize rows to unit length
    row_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-10
    U_normalized = U / row_norms

    # Step 5: K-means in eigenspace
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(U_normalized)
    return labels, eigenvalues, U_normalized

if __name__ == "__main__":
    # Run on two moons
    labels, evals, embedding = spectral_clustering(X, k=2, sigma=0.3)
    accuracy = max(np.mean(labels == y_true), np.mean(labels != y_true))
    print(f"Spectral clustering accuracy: {accuracy:.1%}")

    # Compare with k-means on raw data
    km_labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X)
    km_acc = max(np.mean(km_labels == y_true), np.mean(km_labels != y_true))
    print(f"K-means accuracy:             {km_acc:.1%}")
