"""
Minimum Redundancy Maximum Relevance (mRMR) Selection

Greedy feature selection that maximizes relevance to target while
minimizing redundancy with already-selected features.

Blog post: https://dadops.dev/blog/feature-selection-from-scratch/
"""
import numpy as np
from dataset import make_dataset

X, y, names, n = make_dataset()


def estimate_mi(x, y_or_x2, bins=10):
    """MI between two arrays (continuous-binary or continuous-continuous)."""
    unique_vals = np.unique(y_or_x2)
    if len(unique_vals) <= 10:  # treat as discrete
        mi = 0.0
        c, edges = np.histogram(x, bins=bins)
        for label in unique_vals:
            mask = y_or_x2 == label
            p_y = mask.mean()
            if p_y == 0:
                continue
            c_xy, _ = np.histogram(x[mask], bins=edges)
            for b in range(bins):
                p_x = c[b] / len(x)
                p_xy = c_xy[b] / len(x)
                if p_xy > 0 and p_x > 0:
                    mi += p_xy * np.log2(p_xy / (p_x * p_y))
        return mi
    else:  # continuous-continuous: 2D histogram
        h_xy, _, _ = np.histogram2d(x, y_or_x2, bins=bins)
        h_xy = h_xy / h_xy.sum()
        h_x = h_xy.sum(axis=1)
        h_y = h_xy.sum(axis=0)
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if h_xy[i, j] > 0 and h_x[i] > 0 and h_y[j] > 0:
                    mi += h_xy[i, j] * np.log2(h_xy[i, j] / (h_x[i] * h_y[j]))
        return mi


def mrmr_select(X, y, n_features=5, bins=10):
    """Greedy mRMR: max relevance - mean redundancy, using MI."""
    d = X.shape[1]
    relevance = np.array([estimate_mi(X[:, j], y, bins) for j in range(d)])
    selected = []
    remaining = list(range(d))

    for k in range(n_features):
        best_score, best_f = -np.inf, None
        for f in remaining:
            if len(selected) == 0:
                redundancy = 0.0
            else:
                redundancy = np.mean([
                    estimate_mi(X[:, f], X[:, s], bins)
                    for s in selected
                ])
            score = relevance[f] - redundancy
            if score > best_score:
                best_score, best_f = score, f
        selected.append(best_f)
        remaining.remove(best_f)
    return selected


naive_top5 = np.argsort(-np.array([estimate_mi(X[:, j], y) for j in range(10)]))[:5]
mrmr_top5 = mrmr_select(X, y, n_features=5)

print("Naive MI top-5: ", [names[i] for i in naive_top5])
print("mRMR top-5:     ", [names[i] for i in mrmr_top5])
