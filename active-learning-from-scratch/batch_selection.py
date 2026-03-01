"""
Batch active learning: top-k vs k-centers vs hybrid selection.

Compares three batch selection strategies:
  - Top-K: greedy, picks highest acquisition scores (redundant)
  - K-Centers: maximizes coverage (ignores uncertainty)
  - Hybrid: balances uncertainty and diversity

Requires: numpy, scipy, scikit-learn

From: https://dadops.dev/blog/active-learning-from-scratch/
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def select_batch_topk(scores, k):
    """Naive: take top-k highest scores."""
    return np.argsort(scores)[-k:]

def select_batch_kcenter(X_unlabeled, X_labeled, k):
    """Greedy k-centers: maximize coverage."""
    selected = []
    for _ in range(k):
        if len(selected) == 0:
            ref_points = X_labeled
        else:
            ref_points = np.vstack([X_labeled, X_unlabeled[selected]])
        dists = cdist(X_unlabeled, ref_points).min(axis=1)
        dists[selected] = -1  # exclude already selected
        selected.append(np.argmax(dists))
    return np.array(selected)

def select_batch_hybrid(X_unlabeled, X_labeled, scores, k, alpha=0.5):
    """Uncertainty x diversity: best of both worlds."""
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    selected = []
    for _ in range(k):
        if len(selected) == 0:
            ref_points = X_labeled
        else:
            ref_points = np.vstack([X_labeled, X_unlabeled[selected]])
        dists = cdist(X_unlabeled, ref_points).min(axis=1)
        norm_dists = (dists - dists.min()) / (dists.max() - dists.min() + 1e-10)
        combined = alpha * norm_scores + (1 - alpha) * norm_dists
        combined[selected] = -1
        selected.append(np.argmax(combined))
    return np.array(selected)


if __name__ == "__main__":
    np.random.seed(42)

    # Generate 2D data with 3 clusters
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=42)

    # Small seed set
    seed_idx = []
    for c in [0, 1]:
        class_idx = np.where(y == c)[0][:3]
        seed_idx.extend(class_idx)
    X_seed, y_seed = X[seed_idx], y[seed_idx]

    # Train model to get uncertainty scores
    model = LogisticRegression().fit(X_seed, y_seed)
    probs = model.predict_proba(X)
    uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    batch_size = 10
    print(f"Pool size: {len(X)}, Batch size: {batch_size}\n")

    # Compare strategies
    topk_idx = select_batch_topk(uncertainty, batch_size)
    kcenter_idx = select_batch_kcenter(X, X_seed, batch_size)
    hybrid_idx = select_batch_hybrid(X, X_seed, uncertainty, batch_size)

    def batch_stats(name, idx):
        """Compute spread (avg pairwise distance) and avg uncertainty."""
        pairwise = cdist(X[idx], X[idx])
        spread = pairwise[np.triu_indices(len(idx), k=1)].mean()
        avg_unc = uncertainty[idx].mean()
        print(f"{name:15s}: avg_uncertainty={avg_unc:.4f}, spread={spread:.4f}")

    batch_stats("Top-K", topk_idx)
    batch_stats("K-Centers", kcenter_idx)
    batch_stats("Hybrid", hybrid_idx)
