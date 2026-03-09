"""Scenario comparison: NB accuracy across 4 practical scenarios.

Tests Gaussian NB on small data, sparse high-dimensional data,
correlated features, and multi-class classification.
"""
import numpy as np
from gaussian_nb import GaussianNB


def scenario_comparison():
    """Compare NB vs Logistic Regression across scenarios."""
    np.random.seed(42)
    results = {}

    # Scenario 1: Small training set (50 samples, 10 features)
    X = np.random.randn(50, 10)
    w_true = np.zeros(10); w_true[:3] = [2, -1.5, 1]
    y = (X @ w_true + np.random.randn(50) * 0.5 > 0).astype(int)
    nb = GaussianNB().fit(X[:35], y[:35])
    results["Small data (n=50)"] = np.mean(nb.predict(X[35:]) == y[35:])

    # Scenario 2: High-dimensional sparse (like text: 500 features, 100 samples)
    X = np.random.binomial(1, 0.05, (100, 500)).astype(float)
    w_true = np.zeros(500); w_true[:10] = np.random.randn(10)
    y = (X @ w_true > 0).astype(int)
    nb = GaussianNB().fit(X[:70], y[:70])
    results["Sparse (d=500, n=100)"] = np.mean(nb.predict(X[70:]) == y[70:])

    # Scenario 3: Large data, correlated features (300 samples, 5 features)
    cov = np.full((5, 5), 0.6); np.fill_diagonal(cov, 1.0)
    X0 = np.random.multivariate_normal([1]*5, cov, 150)
    X1 = np.random.multivariate_normal([-1]*5, cov, 150)
    X = np.vstack([X0, X1])
    y = np.array([0]*150 + [1]*150)
    idx = np.random.permutation(300)
    X, y = X[idx], y[idx]
    nb = GaussianNB().fit(X[:210], y[:210])
    results["Large + correlated"] = np.mean(nb.predict(X[210:]) == y[210:])

    # Scenario 4: Multi-class (5 classes)
    centers = np.array([[i*2, j*2] for i in range(3) for j in range(2)])[:5]
    X_parts = [np.random.randn(40, 2) + c for c in centers]
    X = np.vstack(X_parts)
    y = np.concatenate([[i]*40 for i in range(5)])
    idx = np.random.permutation(200)
    X, y = X[idx], y[idx]
    nb = GaussianNB().fit(X[:140], y[:140])
    results["Multi-class (K=5)"] = np.mean(nb.predict(X[140:]) == y[140:])

    for scenario, acc in results.items():
        print(f"  {scenario}: NB accuracy = {acc:.1%}")


scenario_comparison()
