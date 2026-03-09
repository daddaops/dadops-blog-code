import numpy as np

def kmeans(X, K, max_iters=100, seed=42):
    """K-Means clustering with random initialization."""
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]

    # Step 1: randomly pick K data points as initial centroids
    indices = rng.choice(n_samples, K, replace=False)
    centroids = X[indices].copy()

    for _ in range(max_iters):
        # Step 2: assign each point to nearest centroid
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 3: recompute centroids as cluster means
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0
            else centroids[k]
            for k in range(K)
        ])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    # Compute WCSS (within-cluster sum of squares)
    wcss = sum(np.sum((X[labels == k] - centroids[k]) ** 2)
               for k in range(K))
    return labels, centroids, wcss

def make_data():
    """Generate 3 well-separated clusters."""
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) + np.array([0, 5])
    cluster2 = np.random.randn(50, 2) + np.array([5, 0])
    cluster3 = np.random.randn(50, 2) + np.array([-5, 0])
    return np.vstack([cluster1, cluster2, cluster3])

if __name__ == "__main__":
    # Generate 3 well-separated clusters
    X = make_data()

    labels, centroids, wcss = kmeans(X, K=3)
    print(f"Converged! WCSS = {wcss:.1f}")
    print(f"Centroid 1: [{centroids[0, 0]:.2f}, {centroids[0, 1]:.2f}]")
    print(f"Centroid 2: [{centroids[1, 0]:.2f}, {centroids[1, 1]:.2f}]")
    print(f"Centroid 3: [{centroids[2, 0]:.2f}, {centroids[2, 1]:.2f}]")
    # Converged! WCSS = 279.4
    # Centroid 1: [0.06, 5.07]
    # Centroid 2: [4.88, -0.07]
    # Centroid 3: [-4.90, 0.14]
