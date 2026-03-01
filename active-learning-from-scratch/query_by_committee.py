"""
Query-by-Committee (QBC): active learning with model disagreement.

Trains a committee of bootstrapped logistic regression models and
queries examples where vote entropy is highest.

Requires: numpy, scikit-learn

From: https://dadops.dev/blog/active-learning-from-scratch/
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.datasets import make_classification

def query_by_committee(X_pool, y_pool, X_train, y_train,
                       n_committee=5, n_queries=50):
    pool_mask = np.ones(len(X_pool), dtype=bool)
    accuracies = []

    for _ in range(n_queries):
        # Train committee on bootstrap samples
        committee = []
        for _ in range(n_committee):
            # Resample until we have both classes (small seed sets can produce single-class boots)
            for _attempt in range(100):
                X_boot, y_boot = resample(X_train, y_train)
                if len(np.unique(y_boot)) >= 2:
                    break
            model = LogisticRegression().fit(X_boot, y_boot)
            committee.append(model)

        # Evaluate with full committee
        full_model = LogisticRegression().fit(X_train, y_train)
        accuracies.append(full_model.score(X_pool, y_pool))

        # Measure disagreement via vote entropy
        X_unlabeled = X_pool[pool_mask]
        votes = np.array([m.predict(X_unlabeled) for m in committee])
        n_classes = len(np.unique(y_pool))
        vote_entropy = np.zeros(len(X_unlabeled))
        for i in range(len(X_unlabeled)):
            counts = np.bincount(votes[:, i], minlength=n_classes)
            freqs = counts / n_committee
            vote_entropy[i] = -np.sum(freqs * np.log(freqs + 1e-10))

        query_idx = np.where(pool_mask)[0][np.argmax(vote_entropy)]
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
    print(f"Pool size: {len(X)}, Seed size: {len(X_seed)}, Queries: {n_queries}")
    print(f"Committee size: 5\n")

    accs = query_by_committee(X, y, X_seed.copy(), y_seed.copy(), n_queries=n_queries)
    print(f"QBC: start={accs[0]:.3f}, end={accs[-1]:.3f}, avg={np.mean(accs):.3f}")
