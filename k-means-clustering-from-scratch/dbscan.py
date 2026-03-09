import numpy as np
from kmeans_pp import kmeans_pp

def dbscan(X, eps=0.5, min_pts=5):
    """DBSCAN: density-based clustering with noise detection."""
    n = len(X)
    labels = np.full(n, -1)  # -1 = unvisited
    cluster_id = 0

    # precompute pairwise distances
    dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)

    for i in range(n):
        if labels[i] != -1:
            continue  # already assigned

        # find neighbors within eps
        neighbors = np.where(dists[i] <= eps)[0]

        if len(neighbors) < min_pts:
            labels[i] = -2  # noise point
            continue

        # start a new cluster
        labels[i] = cluster_id
        seed_set = list(neighbors)
        idx = 0

        while idx < len(seed_set):
            q = seed_set[idx]
            if labels[q] == -2:
                labels[q] = cluster_id  # noise becomes border point
            if labels[q] != -1:
                idx += 1
                continue

            labels[q] = cluster_id
            q_neighbors = np.where(dists[q] <= eps)[0]
            if len(q_neighbors) >= min_pts:
                seed_set.extend(q_neighbors.tolist())
            idx += 1

        cluster_id += 1

    return labels

if __name__ == "__main__":
    # Two moon-shaped clusters
    np.random.seed(42)
    n = 150
    angles_top = np.linspace(0, np.pi, n)
    angles_bot = np.linspace(0, np.pi, n)
    moon1 = np.column_stack([np.cos(angles_top), np.sin(angles_top)])
    moon2 = np.column_stack([1 - np.cos(angles_bot), 0.5 - np.sin(angles_bot)])
    moon1 += np.random.randn(n, 2) * 0.08
    moon2 += np.random.randn(n, 2) * 0.08
    X_moons = np.vstack([moon1, moon2])

    # K-Means fails: splits moons incorrectly
    km_labels, _, _ = kmeans_pp(X_moons, K=2)
    km_acc = max(np.mean(km_labels[:n] == km_labels[0]),
                 np.mean(km_labels[:n] != km_labels[0]))

    # DBSCAN succeeds: follows the density
    db_labels = dbscan(X_moons, eps=0.2, min_pts=5)
    n_clusters = len(set(db_labels)) - (1 if -2 in db_labels else 0)
    n_noise = np.sum(db_labels == -2)

    print(f"K-Means: cluster purity = {km_acc:.0%}")
    print(f"DBSCAN:  {n_clusters} clusters found, {n_noise} noise points")
    # K-Means: cluster purity = 78%
    # DBSCAN:  2 clusters found, 3 noise points
