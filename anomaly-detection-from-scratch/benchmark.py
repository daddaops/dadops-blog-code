"""Anomaly Detection Benchmark: Four Methods, Three Scenarios.

Compares z-score (distance from centroid) across three data scenarios:
1. Gaussian blob + global outliers (z-score's strength)
2. Multi-density clusters + local outliers (z-score's weakness)
3. High-dimensional data with subtle anomalies

Demonstrates why different scenarios call for different methods.
"""
import random
import math

def benchmark(data, labels, method_scores, method_name, top_k=None):
    """Compute precision and recall for a scored anomaly detector."""
    n_anomalies = sum(labels)
    if top_k is None:
        top_k = n_anomalies  # flag as many as there are true anomalies
    ranked = sorted(range(len(data)), key=lambda i: method_scores[i], reverse=True)
    flagged = ranked[:top_k]
    tp = sum(1 for i in flagged if labels[i] == 1)
    precision = tp / top_k if top_k > 0 else 0
    recall = tp / n_anomalies if n_anomalies > 0 else 0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
    return {"method": method_name, "P": precision, "R": recall, "F1": f1}

if __name__ == "__main__":
    # Scenario 1: Gaussian blob + global outliers (z-score's home turf)
    random.seed(1)
    s1_data = [(random.gauss(0,1), random.gauss(0,1)) for _ in range(200)]
    s1_data += [(random.uniform(5,8), random.uniform(5,8)) for _ in range(8)]
    s1_labels = [0]*200 + [1]*8

    # Scenario 2: Two clusters, very different densities + local outliers
    random.seed(2)
    s2_data = [(random.gauss(0,0.2), random.gauss(0,0.2)) for _ in range(100)]
    s2_data += [(random.gauss(5,1.5), random.gauss(5,1.5)) for _ in range(80)]
    s2_data += [(1.5, 1.5), (3.0, 3.0), (-0.8, 2.0)]  # edge/gap anomalies
    s2_labels = [0]*180 + [1]*3

    # Scenario 3: 8-dimensional data with subtle anomalies
    random.seed(3)
    s3_data = [[random.gauss(0,1) for _ in range(8)] for _ in range(300)]
    s3_data += [[random.gauss(0,1) + 2.5*((j%3)==0) for j in range(8)]
                 for _ in range(6)]  # shifted in only 3 of 8 dimensions
    s3_labels = [0]*300 + [1]*6

    # Run simplified benchmarks (using z-score on flattened distance-from-mean)
    for name, data, labels in [("Gaussian+Global", s1_data, s1_labels),
                                ("Multi-Density", s2_data, s2_labels),
                                ("High-Dim Subtle", s3_data, s3_labels)]:
        # Z-score: distance from centroid
        centroid = [sum(p[d] for p in data)/len(data) for d in range(len(data[0]))]
        z_scores = [math.sqrt(sum((p[d]-centroid[d])**2
                    for d in range(len(p)))) for p in data]
        print(f"\n--- {name} ---")
        print(benchmark(data, labels, z_scores, "Z-score"))
