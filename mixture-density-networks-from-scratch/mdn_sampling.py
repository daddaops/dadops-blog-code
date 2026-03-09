"""Sampling from a Gaussian mixture model.

Demonstrates the two-step sampling process: choose a component
from the categorical distribution, then sample from that Gaussian.
"""
import numpy as np

def sample_mdn(pi, mu, sigma, n_samples=100):
    """Draw samples from a Gaussian mixture."""
    samples = []
    for _ in range(n_samples):
        # Step 1: choose component k based on mixing weights
        k = np.random.choice(len(pi), p=pi)
        # Step 2: sample from that Gaussian
        sample = np.random.normal(mu[k], sigma[k])
        samples.append(sample)
    return np.array(samples)

# Suppose our trained MDN predicts these params for y*=0.5:
pi_pred = np.array([0.33, 0.35, 0.32])       # roughly equal modes
mu_pred = np.array([0.19, 0.50, 0.81])       # three solutions
sigma_pred = np.array([0.025, 0.025, 0.025]) # tight around each

samples = sample_mdn(pi_pred, mu_pred, sigma_pred, n_samples=300)
print(f"Sample mean: {samples.mean():.3f}")   # ~0.50 (misleading!)
print(f"Sample std:  {samples.std():.3f}")    # ~0.26 (high variance)

# But look at the histogram — three clear peaks at 0.19, 0.50, 0.81
for lo, hi, label in [(0.0, 0.35, "Mode 1"), (0.35, 0.65, "Mode 2"),
                       (0.65, 1.0, "Mode 3")]:
    count = ((samples >= lo) & (samples < hi)).sum()
    print(f"  {label} ({lo:.2f}-{hi:.2f}): {count} samples")
