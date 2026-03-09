"""Build three types of similarity graphs from two-moons data."""
import numpy as np
from sklearn.datasets import make_moons

X, y_true = make_moons(n_samples=200, noise=0.06, random_state=42)

def rbf_similarity(X, sigma=0.3):
    """Fully connected similarity matrix with RBF kernel."""
    n = X.shape[0]
    dists_sq = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    return np.exp(-dists_sq / (2 * sigma ** 2))

def knn_graph(X, k=10, sigma=0.3):
    """k-nearest neighbor graph (symmetrized)."""
    W_full = rbf_similarity(X, sigma)
    n = W_full.shape[0]
    W_knn = np.zeros_like(W_full)
    for i in range(n):
        neighbors = np.argsort(W_full[i])[-k-1:-1]  # top-k excluding self
        W_knn[i, neighbors] = W_full[i, neighbors]
    return np.maximum(W_knn, W_knn.T)  # symmetrize

def epsilon_graph(X, epsilon=0.5, sigma=0.3):
    """Epsilon-neighborhood graph."""
    dists_sq = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    mask = dists_sq < epsilon ** 2
    np.fill_diagonal(mask, False)
    W_full = rbf_similarity(X, sigma)
    return W_full * mask

if __name__ == "__main__":
    # Build all three
    W_full = rbf_similarity(X, sigma=0.3)
    W_knn = knn_graph(X, k=10, sigma=0.3)
    W_eps = epsilon_graph(X, epsilon=0.4, sigma=0.3)
    print(f"Fully connected: {np.count_nonzero(W_full):.0f} nonzero entries")
    print(f"KNN (k=10):      {np.count_nonzero(W_knn):.0f} nonzero entries")
    print(f"Epsilon (e=0.4): {np.count_nonzero(W_eps):.0f} nonzero entries")
