import numpy as np
from kmeans_basic import kmeans, make_data

def kmeans_plus_plus_init(X, K, rng):
    """K-Means++ initialization: spread centroids apart."""
    n_samples = X.shape[0]
    centroids = [X[rng.randint(n_samples)]]

    for _ in range(1, K):
        # compute distance from each point to nearest centroid
        dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        # probability proportional to D(x)^2
        probs = dists / dists.sum()
        centroids.append(X[rng.choice(n_samples, p=probs)])

    return np.array(centroids)

def kmeans_pp(X, K, max_iters=100, seed=42):
    """K-Means with K-Means++ initialization."""
    rng = np.random.RandomState(seed)
    centroids = kmeans_plus_plus_init(X, K, rng)

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0
            else centroids[k]
            for k in range(K)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    wcss = sum(np.sum((X[labels == k] - centroids[k]) ** 2) for k in range(K))
    return labels, centroids, wcss

if __name__ == "__main__":
    X = make_data()

    # Compare: random init vs K-Means++ across 20 runs
    random_scores, pp_scores = [], []
    for seed in range(20):
        _, _, w_rand = kmeans(X, K=3, seed=seed)
        _, _, w_pp = kmeans_pp(X, K=3, seed=seed)
        random_scores.append(w_rand)
        pp_scores.append(w_pp)

    print(f"Random init: mean WCSS = {np.mean(random_scores):.1f} ± {np.std(random_scores):.1f}")
    print(f"K-Means++:   mean WCSS = {np.mean(pp_scores):.1f} ± {np.std(pp_scores):.1f}")
    # Random init: mean WCSS = 289.3 ± 15.7
    # K-Means++:   mean WCSS = 279.4 ± 0.0
