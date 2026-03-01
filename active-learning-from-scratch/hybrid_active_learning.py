"""
Hybrid active learning: epsilon-greedy to avoid sampling bias.

Demonstrates how pure uncertainty sampling can develop blind spots
(sampling bias), and how mixing in random exploration fixes it.

Requires: numpy, scikit-learn

From: https://dadops.dev/blog/active-learning-from-scratch/
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def entropy_sampling(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

def hybrid_active_learning(X_pool, y_pool, X_seed, y_seed,
                           n_queries=50, epsilon=0.15):
    """Active learning with random exploration to avoid sampling bias."""
    X_train, y_train = X_seed.copy(), y_seed.copy()
    pool_mask = np.ones(len(X_pool), dtype=bool)
    accuracies = []
    rng = np.random.RandomState(42)

    for _ in range(n_queries):
        model = LogisticRegression().fit(X_train, y_train)
        accuracies.append(model.score(X_pool, y_pool))

        pool_indices = np.where(pool_mask)[0]
        if rng.random() < epsilon:
            # Explore: random query
            query_idx = rng.choice(pool_indices)
        else:
            # Exploit: uncertainty query
            probs = model.predict_proba(X_pool[pool_indices])
            uncertainty = entropy_sampling(probs)
            query_idx = pool_indices[np.argmax(uncertainty)]

        X_train = np.vstack([X_train, X_pool[query_idx:query_idx+1]])
        y_train = np.append(y_train, y_pool[query_idx])
        pool_mask[query_idx] = False

    return accuracies

def pure_uncertainty_loop(X_pool, y_pool, X_seed, y_seed, n_queries=50):
    """Pure uncertainty sampling (no exploration)."""
    X_train, y_train = X_seed.copy(), y_seed.copy()
    pool_mask = np.ones(len(X_pool), dtype=bool)
    accuracies = []

    for _ in range(n_queries):
        model = LogisticRegression().fit(X_train, y_train)
        accuracies.append(model.score(X_pool, y_pool))

        pool_indices = np.where(pool_mask)[0]
        probs = model.predict_proba(X_pool[pool_indices])
        uncertainty = entropy_sampling(probs)
        query_idx = pool_indices[np.argmax(uncertainty)]

        X_train = np.vstack([X_train, X_pool[query_idx:query_idx+1]])
        y_train = np.append(y_train, y_pool[query_idx])
        pool_mask[query_idx] = False

    return accuracies


if __name__ == "__main__":
    np.random.seed(42)

    # Generate 2D binary classification data
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               random_state=42)

    # Small seed set
    seed_idx = []
    for c in [0, 1]:
        class_idx = np.where(y == c)[0][:2]
        seed_idx.extend(class_idx)
    X_seed, y_seed = X[seed_idx], y[seed_idx]

    n_queries = 50
    print(f"Pool size: {len(X)}, Seed size: {len(X_seed)}, Queries: {n_queries}\n")

    pure_accs = pure_uncertainty_loop(X, y, X_seed.copy(), y_seed.copy(),
                                      n_queries=n_queries)
    hybrid_accs = hybrid_active_learning(X, y, X_seed.copy(), y_seed.copy(),
                                          n_queries=n_queries, epsilon=0.15)

    print(f"{'Pure Uncertainty':20s}: start={pure_accs[0]:.3f}, "
          f"end={pure_accs[-1]:.3f}, avg={np.mean(pure_accs):.3f}")
    print(f"{'Hybrid (eps=0.15)':20s}: start={hybrid_accs[0]:.3f}, "
          f"end={hybrid_accs[-1]:.3f}, avg={np.mean(hybrid_accs):.3f}")
