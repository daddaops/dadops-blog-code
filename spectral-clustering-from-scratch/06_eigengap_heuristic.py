"""Use the eigengap heuristic to estimate the number of clusters."""
import numpy as np
from sklearn.datasets import make_moons, make_circles

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

if __name__ == "__main__":
    datasets = {
        "2 clusters (moons)": make_moons(200, noise=0.06, random_state=42)[0],
        "3 clusters (blobs)": np.vstack([
            np.random.RandomState(42).randn(70, 2) * 0.4 + center
            for center in [(-2, 0), (2, 0), (0, 3)]
        ]),
        "2 clusters (circles)": make_circles(200, noise=0.05, factor=0.4,
                                              random_state=42)[0],
    }

    for name, data in datasets.items():
        W = rbf_similarity(data, sigma=0.5)
        L = compute_laplacian(W, normalized=False)
        evals = np.linalg.eigvalsh(L)

        # Find eigengap
        gaps = np.diff(evals[:10])
        k_est = np.argmax(gaps) + 1

        print(f"{name}")
        print(f"  First 8 eigenvalues: {np.round(evals[:8], 4)}")
        print(f"  Largest gap at k={k_est} "
              f"(gap={gaps[k_est-1]:.4f})")
        print()
