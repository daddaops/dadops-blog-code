"""Evaluation splitting strategies: random, stratified, temporal.

Demonstrates three data-splitting approaches and the variance problem
of single holdout evaluation across 20 random seeds.
"""
import numpy as np

def random_holdout(X, y, test_ratio=0.2, seed=42):
    """Split data randomly into train and test sets."""
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    return (X[indices[:split]], X[indices[split:]],
            y[indices[:split]], y[indices[split:]])

def stratified_holdout(X, y, test_ratio=0.2, seed=42):
    """Split preserving class proportions in both sets."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        split = int(len(label_idx) * (1 - test_ratio))
        train_idx.extend(label_idx[:split])
        test_idx.extend(label_idx[split:])
    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx])

def temporal_split(X, y, timestamps, cutoff):
    """Split by time: train on past, test on future."""
    train_mask = timestamps < cutoff
    test_mask = timestamps >= cutoff
    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask])

# Demonstrate the variance problem: 20 random splits
# Using a simple logistic regression on synthetic data
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
n_samples = 500
X = np.random.randn(n_samples, 10)
w = np.random.randn(10)
y = (X @ w + np.random.randn(n_samples) * 2 > 0).astype(int)

accuracies = []
for seed in range(20):
    X_tr, X_te, y_tr, y_te = random_holdout(X, y, seed=seed)
    model = LogisticRegression(max_iter=200)
    model.fit(X_tr, y_tr)
    accuracies.append(model.score(X_te, y_te))
print(f"Mean: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
# Mean: 0.912 +/- 0.031  — that +/-3.1% is the problem!
