import numpy as np
from lance_williams import hierarchical_cluster

def find_clusters(Z, n, n_clusters):
    """Extract flat cluster labels by cutting dendrogram."""
    labels = np.zeros(n, dtype=int)
    members = {i: [i] for i in range(n)}
    for i, (c1, c2, d, _) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        members[n + i] = members.get(c1, [c1]) + members.get(c2, [c2])

    # Cut: use the last (n_clusters - 1) merges as boundaries
    root = n + len(Z) - 1
    clusters = [root]
    for _ in range(n_clusters - 1):
        # Split the cluster merged at highest distance
        highest = max(clusters, key=lambda c: Z[c - n][2] if c >= n else -1)
        if highest < n:
            break
        row = Z[highest - n]
        clusters.remove(highest)
        clusters.extend([int(row[0]), int(row[1])])

    for label, cid in enumerate(clusters):
        for pt in members.get(cid, [cid]):
            labels[pt] = label
    return labels

def inconsistency_cut(Z):
    """Find best cut using inconsistency in merge distances."""
    heights = Z[:, 2]
    # Compute gaps between consecutive merge heights
    gaps = np.diff(heights)
    if len(gaps) == 0:
        return 1
    best_gap_idx = np.argmax(gaps)
    # Number of clusters = n - (index of gap) - 1
    n_clusters = len(Z) - best_gap_idx
    return n_clusters

if __name__ == "__main__":
    # Demo: find natural clusters
    np.random.seed(7)
    X = np.vstack([np.random.randn(15, 2) * 0.5 + [0, 0],
                   np.random.randn(15, 2) * 0.5 + [4, 4],
                   np.random.randn(15, 2) * 0.5 + [8, 1]])
    Z = hierarchical_cluster(X, "ward")
    k = inconsistency_cut(Z)
    labels = find_clusters(Z, len(X), k)
    print(f"Suggested clusters: {k}")
    print(f"Cluster sizes: {[np.sum(labels == i) for i in range(k)]}")
