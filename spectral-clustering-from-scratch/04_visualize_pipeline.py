"""Visualize the spectral clustering pipeline: data, embedding, and result."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

X, y_true = make_moons(n_samples=200, noise=0.06, random_state=42)

def rbf_similarity(X, sigma=0.3):
    n = X.shape[0]
    dists_sq = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    return np.exp(-dists_sq / (2 * sigma ** 2))

def compute_laplacian(W, normalized=False):
    D = np.diag(W.sum(axis=1))
    L = D - W
    if not normalized:
        return L
    d_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1) + 1e-10))
    return d_inv_sqrt @ L @ d_inv_sqrt

def spectral_clustering(X, k=2, sigma=0.3, normalized=True):
    W = rbf_similarity(X, sigma)
    L = compute_laplacian(W, normalized=normalized)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    U = eigenvectors[:, :k]
    row_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-10
    U_normalized = U / row_norms
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(U_normalized)
    return labels, eigenvalues, U_normalized

if __name__ == "__main__":
    labels, evals, embedding = spectral_clustering(X, k=2, sigma=0.1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original data with true labels
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', s=15)
    axes[0].set_title("Original Data")

    # Eigenvector embedding
    axes[1].scatter(embedding[:, 0], embedding[:, 1], c=y_true, cmap='coolwarm', s=15)
    axes[1].set_title("Eigenvector Embedding")
    axes[1].set_xlabel("Eigenvector 1")
    axes[1].set_ylabel("Eigenvector 2")

    # Spectral clustering result
    axes[2].scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm', s=15)
    axes[2].set_title("Spectral Clustering Result")

    plt.tight_layout()
    out_path = "output/spectral_pipeline.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
