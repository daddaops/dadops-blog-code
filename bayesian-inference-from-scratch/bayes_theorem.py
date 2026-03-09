"""Bayes' Theorem from Scratch — medical test and grid-based coin updating.

Demonstrates:
1. The base-rate fallacy in medical testing
2. Grid-based Bayesian updating for coin bias estimation
"""
import numpy as np

# The Medical Test Problem
# A disease affects 1% of the population
# Test sensitivity (true positive rate): 95%
# Test specificity (true negative rate): 90%
# You test positive. What's the probability you have the disease?

prevalence = 0.01       # P(disease)
sensitivity = 0.95      # P(positive | disease)
specificity = 0.90      # P(negative | no disease)

# Bayes' theorem
p_positive = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
p_disease_given_positive = (sensitivity * prevalence) / p_positive

print(f"P(disease | positive test) = {p_disease_given_positive:.3f}")
# Output: P(disease | positive test) = 0.088

# Only 8.8%! The base rate (1%) dominates. Most positives are false positives.

# Now let's do general Bayesian updating on a grid
# Problem: estimate the bias of a coin from observed flips

theta_grid = np.linspace(0, 1, 1000)   # 1000 possible bias values
prior = np.ones_like(theta_grid)        # uniform prior: all biases equally likely
prior /= prior.sum()                    # normalize

# Observe: H, H, T, H, T (3 heads, 2 tails)
flips = [1, 1, 0, 1, 0]

posterior = prior.copy()
for flip in flips:
    if flip == 1:  # heads
        likelihood = theta_grid         # P(H | theta) = theta
    else:           # tails
        likelihood = 1 - theta_grid     # P(T | theta) = 1 - theta
    posterior *= likelihood
    posterior /= posterior.sum()         # normalize after each update

map_estimate = theta_grid[np.argmax(posterior)]
mean_estimate = np.sum(theta_grid * posterior)
print(f"MAP estimate: {map_estimate:.3f}")   # ~0.600
print(f"Mean estimate: {mean_estimate:.3f}") # ~0.571
