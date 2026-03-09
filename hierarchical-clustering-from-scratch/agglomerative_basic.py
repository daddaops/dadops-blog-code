import numpy as np

def agglomerative_cluster(X):
    """Bottom-up hierarchical clustering. Returns scipy-style linkage matrix."""
    n = len(X)
    # Pairwise Euclidean distances
    dist = np.full((2 * n, 2 * n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            dist[i, j] = dist[j, i] = d

    active = set(range(n))       # clusters still alive
    sizes = {i: 1 for i in range(n)}
    linkage = []

    for step in range(n - 1):
        # Find closest pair among active clusters
        min_d, ci, cj = np.inf, -1, -1
        for i in active:
            for j in active:
                if i < j and dist[i, j] < min_d:
                    min_d, ci, cj = dist[i, j], i, j

        # Create new cluster with id = n + step
        new_id = n + step
        new_size = sizes[ci] + sizes[cj]
        linkage.append([ci, cj, min_d, new_size])
        sizes[new_id] = new_size

        # Update distances using single linkage (min)
        for k in active:
            if k != ci and k != cj:
                dist[new_id, k] = min(dist[ci, k], dist[cj, k])
                dist[k, new_id] = dist[new_id, k]

        active.discard(ci)
        active.discard(cj)
        active.add(new_id)

    return np.array(linkage)

if __name__ == "__main__":
    # Test on simple data
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8],
                  [1, 0.6], [9, 11], [8, 2], [10, 2]])
    Z = agglomerative_cluster(X)
    for row in Z:
        print(f"Merge {int(row[0]):2d} + {int(row[1]):2d}  "
              f"dist={row[2]:.2f}  size={int(row[3])}")
