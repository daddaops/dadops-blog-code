"""k-NN Anomaly Detector: Distance-Based Scoring.

Non-parametric anomaly detection using average distance to k nearest neighbors.
Points far from their neighbors are flagged as anomalous.

Applied to 2D clustered data with injected outliers.
"""
import math
import random

def euclidean_dist(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))

def knn_anomaly_scores(data, k=5):
    """Compute anomaly score as average distance to k nearest neighbors."""
    n = len(data)
    scores = []
    for i in range(n):
        # Compute distances to all other points
        dists = sorted(
            euclidean_dist(data[i], data[j]) for j in range(n) if j != i
        )
        # Score = average distance to k nearest neighbors
        scores.append(sum(dists[:k]) / k)
    return scores

if __name__ == "__main__":
    # Generate 2D data: two clusters + outliers
    random.seed(7)
    cluster1 = [(random.gauss(2, 0.5), random.gauss(2, 0.5)) for _ in range(40)]
    cluster2 = [(random.gauss(7, 0.8), random.gauss(7, 0.8)) for _ in range(40)]
    outliers = [(random.uniform(-2, 11), random.uniform(-2, 11)) for _ in range(5)]
    data = cluster1 + cluster2 + outliers
    labels = [0]*40 + [0]*40 + [1]*5  # 0=normal, 1=anomaly

    scores = knn_anomaly_scores(data, k=5)
    threshold = sorted(scores, reverse=True)[6]  # top ~7% as anomalies
    flagged = [i for i, s in enumerate(scores) if s >= threshold]

    true_pos = sum(1 for i in flagged if labels[i] == 1)
    print(f"Flagged {len(flagged)} points, {true_pos} are true anomalies")
    print(f"Precision: {true_pos/len(flagged):.2f}")
    print(f"Recall:    {true_pos/5:.2f}")
