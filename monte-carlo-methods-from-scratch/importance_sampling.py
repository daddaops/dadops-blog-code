"""Importance sampling for rare event estimation.

Estimates P(X > 4) for X ~ N(0,1) using both naive MC and
importance sampling with a shifted proposal distribution.
"""
import numpy as np
from scipy.stats import norm

np.random.seed(42)

# Goal: estimate P(X > 4) where X ~ N(0, 1)
true_prob = 1 - norm.cdf(4)  # about 3.17e-5

# Naive Monte Carlo: draw from N(0, 1), count exceedances
n_naive = 100_000
samples_naive = np.random.normal(0, 1, n_naive)
naive_estimate = np.mean(samples_naive > 4)
naive_stderr = np.std(samples_naive > 4) / np.sqrt(n_naive)
print(f"True P(X > 4):       {true_prob:.2e}")
print(f"Naive MC (N=100K):    {naive_estimate:.2e} +/- {naive_stderr:.2e}")

# Importance sampling: draw from q = N(4, 1), reweight by p/q
n_is = 1_000
samples_is = np.random.normal(4, 1, n_is)
weights = (samples_is > 4) * norm.pdf(samples_is, 0, 1) / norm.pdf(samples_is, 4, 1)
is_estimate = np.mean(weights)
is_stderr = np.std(weights) / np.sqrt(n_is)
print(f"Importance Sampling (N=1K): {is_estimate:.2e} +/- {is_stderr:.2e}")
