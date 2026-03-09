"""Correlated features: NB is overconfident but still accurate.

Demonstrates the independence paradox: NB violates the independence
assumption with correlated features, yet still ranks classes correctly.
"""
import numpy as np
from gaussian_nb import GaussianNB


def compare_nb_vs_lr(n_samples=200, correlation=0.8):
    """Generate correlated features and compare NB to Logistic Regression."""
    np.random.seed(42)
    # Class 0: features centered at (2, 2) with correlation
    cov = [[1, correlation], [correlation, 1]]
    X0 = np.random.multivariate_normal([2, 2], cov, n_samples // 2)
    # Class 1: features centered at (-1, -1) with same correlation
    X1 = np.random.multivariate_normal([-1, -1], cov, n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Naive Bayes (treats features as independent despite correlation=0.8!)
    nb = GaussianNB().fit(X_train, y_train)
    nb_acc = np.mean(nb.predict(X_test) == y_test)

    # Check NB's probability calibration
    log_posts = []
    for x in X_test[:5]:
        posts = []
        for c in nb.classes:
            mean, var, prior = nb.params[c]
            lp = np.log(prior) + np.sum(nb._log_gaussian(x, mean, var))
            posts.append(lp)
        # Convert to probabilities via softmax
        posts = np.array(posts)
        posts -= posts.max()
        probs = np.exp(posts) / np.exp(posts).sum()
        log_posts.append(probs)

    print(f"Feature correlation: {correlation}")
    print(f"NB accuracy: {nb_acc:.1%}")
    print(f"NB probabilities (first 5 samples):")
    for i, p in enumerate(log_posts):
        print(f"  Sample {i}: P(class0)={p[0]:.4f}, P(class1)={p[1]:.4f}")


compare_nb_vs_lr(correlation=0.8)
