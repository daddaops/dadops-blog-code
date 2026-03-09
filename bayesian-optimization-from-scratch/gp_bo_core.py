"""Core GP and acquisition function components for Bayesian Optimization.

Provides:
- RBF kernel
- GP posterior via Cholesky decomposition
- Expected Improvement, UCB, and Probability of Improvement acquisition functions
"""
import numpy as np
from scipy.stats import norm


def rbf_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
    """Squared exponential kernel — assumes smooth functions."""
    dist_sq = (np.sum(X1**2, axis=1, keepdims=True)
               - 2 * X1 @ X2.T
               + np.sum(X2**2, axis=1))
    return signal_var * np.exp(-0.5 * dist_sq / length_scale**2)


def gp_posterior(X_train, y_train, X_test,
                 length_scale=1.0, signal_var=1.0, noise_var=0.01):
    """GP posterior mean and variance via Cholesky decomposition."""
    K = rbf_kernel(X_train, X_train, length_scale, signal_var)
    K += noise_var * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, length_scale, signal_var)
    K_ss = rbf_kernel(X_test, X_test, length_scale, signal_var)

    L = np.linalg.cholesky(K)                       # Stable factorization
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = K_s.T @ alpha                               # Posterior mean
    v = np.linalg.solve(L, K_s)
    var = np.diag(K_ss - v.T @ v)                    # Posterior variance
    return mu, np.maximum(var, 1e-10)


def expected_improvement(mu, sigma, f_best, xi=0.01):
    """EI: expected amount of improvement over current best."""
    z = (f_best - mu - xi) / sigma
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-10] = 0.0
    return ei


def upper_confidence_bound(mu, sigma, kappa=2.0):
    """UCB: optimistic estimate (lower is better for minimization)."""
    return -mu + kappa * sigma


def probability_of_improvement(mu, sigma, f_best, xi=0.01):
    """PI: probability of beating the current best."""
    z = (f_best - mu - xi) / sigma
    pi = norm.cdf(z)
    pi[sigma < 1e-10] = 0.0
    return pi
