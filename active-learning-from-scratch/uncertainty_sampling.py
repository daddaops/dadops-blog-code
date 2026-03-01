"""
Uncertainty sampling: three acquisition functions for active learning.

Implements least confidence, margin sampling, and entropy sampling,
then runs an active learning loop comparing all three against random
baseline on a 2D synthetic classification dataset.

Requires: numpy, scikit-learn

From: https://dadops.dev/blog/active-learning-from-scratch/
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def least_confidence(probs):
    return 1 - np.max(probs, axis=1)

def margin_sampling(probs):
    sorted_probs = np.sort(probs, axis=1)
    return 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])

def entropy_sampling(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

def active_learning_loop(X_pool, y_pool, X_seed, y_seed,
                         n_queries=50, acquisition_fn=entropy_sampling):
    X_train, y_train = X_seed.copy(), y_seed.copy()
    pool_mask = np.ones(len(X_pool), dtype=bool)
    accuracies = []

    for _ in range(n_queries):
        model = LogisticRegression().fit(X_train, y_train)
        accuracies.append(model.score(X_pool, y_pool))

        # Score the remaining pool
        probs = model.predict_proba(X_pool[pool_mask])
        scores = acquisition_fn(probs)
        query_idx = np.where(pool_mask)[0][np.argmax(scores)]

        # "Label" the queried point and add to training set
        X_train = np.vstack([X_train, X_pool[query_idx:query_idx+1]])
        y_train = np.append(y_train, y_pool[query_idx])
        pool_mask[query_idx] = False

    return accuracies

def random_learning_loop(X_pool, y_pool, X_seed, y_seed, n_queries=50):
    """Baseline: random sampling."""
    rng = np.random.RandomState(42)
    X_train, y_train = X_seed.copy(), y_seed.copy()
    pool_mask = np.ones(len(X_pool), dtype=bool)
    accuracies = []

    for _ in range(n_queries):
        model = LogisticRegression().fit(X_train, y_train)
        accuracies.append(model.score(X_pool, y_pool))

        pool_indices = np.where(pool_mask)[0]
        query_idx = rng.choice(pool_indices)
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

    # Small seed set (4 points, 2 per class)
    seed_idx = []
    for c in [0, 1]:
        class_idx = np.where(y == c)[0][:2]
        seed_idx.extend(class_idx)
    X_seed, y_seed = X[seed_idx], y[seed_idx]

    n_queries = 50
    print(f"Pool size: {len(X)}, Seed size: {len(X_seed)}, Queries: {n_queries}\n")

    for name, fn in [("Least Confidence", least_confidence),
                     ("Margin Sampling", margin_sampling),
                     ("Entropy Sampling", entropy_sampling)]:
        accs = active_learning_loop(X, y, X_seed, y_seed,
                                    n_queries=n_queries, acquisition_fn=fn)
        print(f"{name:20s}: start={accs[0]:.3f}, end={accs[-1]:.3f}, "
              f"avg={np.mean(accs):.3f}")

    random_accs = random_learning_loop(X, y, X_seed, y_seed, n_queries=n_queries)
    print(f"{'Random Baseline':20s}: start={random_accs[0]:.3f}, "
          f"end={random_accs[-1]:.3f}, avg={np.mean(random_accs):.3f}")
