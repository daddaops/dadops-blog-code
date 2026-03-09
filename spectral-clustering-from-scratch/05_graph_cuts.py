"""Compare graph cut metrics between spectral and random partitions."""
import numpy as np
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

def graph_cut_values(W, labels):
    """Compute RatioCut and Ncut for a given partition."""
    unique_labels = np.unique(labels)
    n = len(labels)

    # Cut value: total weight of edges crossing clusters
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                cut += W[i, j]

    # RatioCut: cut/|A| + cut/|B|
    ratiocut = sum(cut / np.sum(labels == lbl) for lbl in unique_labels)

    # Ncut: cut/vol(A) + cut/vol(B)
    volumes = [W[labels == lbl].sum() for lbl in unique_labels]
    ncut = sum(cut / vol for vol in volumes if vol > 0)

    return cut, ratiocut, ncut

if __name__ == "__main__":
    labels, _, _ = spectral_clustering(X, k=2, sigma=0.3)
    W = rbf_similarity(X, sigma=0.3)

    # Spectral clustering partition
    cut_s, rc_s, nc_s = graph_cut_values(W, labels)

    # Random partition for comparison
    rng = np.random.RandomState(0)
    random_labels = rng.randint(0, 2, size=len(X))
    cut_r, rc_r, nc_r = graph_cut_values(W, random_labels)

    print(f"{'Metric':<12} {'Spectral':>10} {'Random':>10}")
    print(f"{'Cut':<12} {cut_s:>10.2f} {cut_r:>10.2f}")
    print(f"{'RatioCut':<12} {rc_s:>10.4f} {rc_r:>10.4f}")
    print(f"{'Ncut':<12} {nc_s:>10.4f} {nc_r:>10.4f}")
