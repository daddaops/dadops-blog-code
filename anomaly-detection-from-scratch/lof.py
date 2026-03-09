"""Local Outlier Factor (LOF) from Scratch.

Density-based anomaly detection that compares each point's local density
to its neighbors' densities. LOF adapts to clusters of different densities,
unlike global distance-based methods.

Applied to dataset with dense and sparse clusters plus true anomalies.
"""
import math
import random

def euclidean(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

def local_outlier_factor(data, k=5):
    """Full LOF implementation: returns anomaly score for each point."""
    n = len(data)

    # Step 1: Compute pairwise distances and k-neighborhoods
    dist_matrix = [[euclidean(data[i], data[j]) for j in range(n)] for i in range(n)]
    k_neighbors = []  # indices of k nearest neighbors per point
    k_distances = []  # distance to k-th nearest neighbor per point
    for i in range(n):
        neighbor_dists = sorted(range(n), key=lambda j: dist_matrix[i][j])
        neighbors = [j for j in neighbor_dists if j != i][:k]
        k_neighbors.append(neighbors)
        k_distances.append(dist_matrix[i][neighbors[-1]])

    # Step 2: Reachability distance — smoothing trick
    def reach_dist(a, b):
        return max(k_distances[b], dist_matrix[a][b])

    # Step 3: Local Reachability Density
    lrd = []
    for i in range(n):
        avg_reach = sum(reach_dist(i, j) for j in k_neighbors[i]) / k
        lrd.append(1.0 / avg_reach if avg_reach > 0 else float('inf'))

    # Step 4: LOF = average neighbor density / own density
    lof_scores = []
    for i in range(n):
        neighbor_avg_lrd = sum(lrd[j] for j in k_neighbors[i]) / k
        lof_scores.append(neighbor_avg_lrd / lrd[i] if lrd[i] > 0 else 0)

    return lof_scores

if __name__ == "__main__":
    # Test on two clusters with VERY different densities
    random.seed(21)
    dense_cluster = [(random.gauss(0, 0.3), random.gauss(0, 0.3)) for _ in range(60)]
    sparse_cluster = [(random.gauss(6, 1.5), random.gauss(6, 1.5)) for _ in range(20)]
    anomalies = [(3.0, 3.0), (-3.0, 4.0), (8.0, -1.0)]  # between/outside clusters
    data = dense_cluster + sparse_cluster + anomalies
    labels = [0]*60 + [0]*20 + [1]*3

    lof = local_outlier_factor(data, k=10)
    top_indices = sorted(range(len(lof)), key=lambda i: lof[i], reverse=True)[:5]
    print("Top 5 LOF scores:")
    for idx in top_indices:
        label = "ANOMALY" if labels[idx] == 1 else "normal"
        print(f"  Point {idx}: LOF={lof[idx]:.2f} ({label})")
