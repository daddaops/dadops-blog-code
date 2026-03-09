"""
K-Means vs. EM Comparison

Direct comparison of k-means and EM/GMM on elliptical clusters.
Shows EM's superior accuracy when clusters have non-spherical shapes.

Blog post: https://dadops.dev/blog/expectation-maximization-from-scratch/
"""
import numpy as np
from itertools import permutations
from gmm_data import make_gmm_data
from em_full import fit_gmm


def run_kmeans(X, K, seed=0, max_iter=50):
    """Standard k-means for comparison."""
    rng = np.random.RandomState(seed)
    centroids = X[rng.choice(len(X), K, replace=False)]
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels


def best_accuracy(pred, true, K):
    return max(np.mean(pred == np.array(p)[true]) for p in permutations(range(K)))


if __name__ == "__main__":
    X, z_true, true_w, true_mu, true_cov = make_gmm_data()

    # Compare on elliptical cluster dataset
    km_labels = run_kmeans(X, 3, seed=7)
    w, mu, cov, gamma, lls = fit_gmm(X, K=3, seed=7)
    em_labels = np.argmax(gamma, axis=1)

    km_acc = best_accuracy(km_labels, z_true, 3)
    em_acc = best_accuracy(em_labels, z_true, 3)
    print(f"K-means accuracy: {km_acc:.1%}")
    print(f"EM/GMM  accuracy: {em_acc:.1%}")
