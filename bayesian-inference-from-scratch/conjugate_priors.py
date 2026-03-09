"""Conjugate Prior Updates — Beta-Binomial and Normal-Normal.

Demonstrates how conjugate priors give analytical posterior updates
and how posteriors sharpen with more data.
"""
import numpy as np

# === Beta-Binomial Conjugacy ===
# Prior: Beta(alpha, beta) — encodes belief about coin bias
# Likelihood: Binomial — counting heads and tails
# Posterior: Beta(alpha + heads, beta + tails)

def beta_binomial_update(alpha_prior, beta_prior, n_heads, n_tails):
    """Update Beta prior with binomial observations."""
    return alpha_prior + n_heads, beta_prior + n_tails

# Start with uniform prior Beta(1, 1)
alpha, bet = 1.0, 1.0

# Simulate observing a biased coin (true bias = 0.7)
np.random.seed(42)
data_sizes = [10, 100, 1000]
for n in data_sizes:
    flips = np.random.binomial(1, 0.7, n)
    a_post, b_post = beta_binomial_update(alpha, bet, flips.sum(), n - flips.sum())
    mean_post = a_post / (a_post + b_post)
    std_post = np.sqrt(a_post * b_post / ((a_post + b_post)**2 * (a_post + b_post + 1)))
    print(f"n={n:4d}: posterior Beta({a_post:.0f}, {b_post:.0f}), "
          f"mean={mean_post:.3f} +/- {std_post:.3f}")
# n=  10: posterior Beta(9, 4),    mean=0.692 +/- 0.124
# n= 100: posterior Beta(74, 29),  mean=0.718 +/- 0.044
# n=1000: posterior Beta(691, 311), mean=0.690 +/- 0.015

# === Normal-Normal Conjugacy ===
# Prior: N(mu_0, sigma_0^2) on the mean
# Likelihood: N(mu, sigma^2) with known variance
# Posterior: precision-weighted average

def normal_normal_update(mu_0, sigma_0, data, sigma_known):
    """Update Normal prior with Gaussian observations."""
    n = len(data)
    precision_prior = 1 / sigma_0**2
    precision_data = n / sigma_known**2
    precision_post = precision_prior + precision_data
    mu_post = (precision_prior * mu_0 + precision_data * np.mean(data)) / precision_post
    sigma_post = np.sqrt(1 / precision_post)
    return mu_post, sigma_post

data = np.random.normal(5.0, 2.0, 50)  # true mean = 5.0
mu_post, sigma_post = normal_normal_update(0.0, 10.0, data, 2.0)
print(f"\nNormal-Normal: posterior mean={mu_post:.2f}, std={sigma_post:.2f}")
# Posterior concentrates near 5.0, much tighter than prior std of 10.0
