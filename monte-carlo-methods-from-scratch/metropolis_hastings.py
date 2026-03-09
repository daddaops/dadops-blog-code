"""Metropolis-Hastings MCMC for sampling a bimodal distribution.

Implements a Gaussian random-walk MH sampler targeting a mixture
of two Gaussians, demonstrating burn-in and acceptance rate.
"""
import numpy as np
from scipy.stats import norm

np.random.seed(42)

# Target: bimodal distribution (known only up to normalizing constant)
def log_target(x):
    return np.log(0.3 * norm.pdf(x, -2, 0.5) + 0.7 * norm.pdf(x, 3, 1.0))

# Metropolis-Hastings with Gaussian random walk
n_samples = 10_000
step_size = 1.5
chain = np.zeros(n_samples)
chain[0] = 0.0
accepted = 0

for i in range(1, n_samples):
    proposal = chain[i-1] + np.random.normal(0, step_size)
    log_alpha = log_target(proposal) - log_target(chain[i-1])

    if np.log(np.random.uniform()) < log_alpha:
        chain[i] = proposal
        accepted += 1
    else:
        chain[i] = chain[i-1]

burn_in = 1000
samples = chain[burn_in:]
print(f"Acceptance rate: {accepted / n_samples:.1%}")
print(f"Sample mean: {np.mean(samples):.3f}")
print(f"Sample std:  {np.std(samples):.3f}")
