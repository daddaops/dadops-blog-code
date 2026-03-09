import numpy as np
from kmeans_basic import make_data
from kmeans_pp import kmeans_pp

def silhouette_score(X, labels):
    """Compute mean silhouette score for a clustering."""
    n = len(X)
    scores = np.zeros(n)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0  # need at least 2 clusters

    for i in range(n):
        # a(i): mean distance to other same-cluster points
        mask = (labels == labels[i])
        mask[i] = False
        same = X[mask]
        a_i = np.mean(np.linalg.norm(same - X[i], axis=1)) if len(same) > 0 else 0

        # b(i): mean distance to nearest other cluster
        b_i = np.inf
        for k in unique_labels:
            if k == labels[i]:
                continue
            other = X[labels == k]
            mean_dist = np.mean(np.linalg.norm(other - X[i], axis=1))
            b_i = min(b_i, mean_dist)

        scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    return np.mean(scores)

def elbow_and_silhouette(X, max_k=10):
    """Compute WCSS and silhouette for K=2..max_k."""
    wcss_vals, sil_vals = [], []
    for k in range(2, max_k + 1):
        labels, _, wcss = kmeans_pp(X, K=k)
        wcss_vals.append(wcss)
        sil_vals.append(silhouette_score(X, labels))
    return wcss_vals, sil_vals

if __name__ == "__main__":
    X = make_data()
    wcss_vals, sil_vals = elbow_and_silhouette(X, max_k=8)
    for k, (w, s) in enumerate(zip(wcss_vals, sil_vals), start=2):
        print(f"K={k}: WCSS={w:>7.1f}, Silhouette={s:.3f}")
    # K=2: WCSS= 586.4, Silhouette=0.541
    # K=3: WCSS= 279.4, Silhouette=0.683
    # K=4: WCSS= 232.8, Silhouette=0.571
    # K=5: WCSS= 196.2, Silhouette=0.508
    # K=6: WCSS= 167.4, Silhouette=0.470
    # K=7: WCSS= 143.9, Silhouette=0.435
    # K=8: WCSS= 124.6, Silhouette=0.418
