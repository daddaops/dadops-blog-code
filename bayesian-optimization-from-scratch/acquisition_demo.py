"""Acquisition Function Demo — compares EI, UCB, and PI suggestions.

Code Block 2 from the blog post.
"""
import numpy as np
from gp_bo_core import (gp_posterior, expected_improvement,
                         upper_confidence_bound, probability_of_improvement)

# Reproduce the GP from Code Block 1
np.random.seed(42)
X_train = np.array([[-0.5], [0.2], [0.8], [1.5], [2.3]])
y_train = (np.sin(2 * X_train.ravel())
           + 0.5 * np.cos(4 * X_train.ravel())
           + 0.1 * np.random.randn(5))

X_test = np.linspace(-1, 3, 200).reshape(-1, 1)
mu, var = gp_posterior(X_train, y_train, X_test, length_scale=0.5)
sigma = np.sqrt(var)

# Apply acquisition functions
f_best = np.min(y_train)  # Current best observation
ei = expected_improvement(mu, sigma, f_best)
ucb = upper_confidence_bound(mu, sigma)
pi = probability_of_improvement(mu, sigma, f_best)

print(f"Current best: f(2.3) = {f_best:.3f}")
print(f"EI suggests:  x = {X_test[np.argmax(ei), 0]:.3f}")
print(f"UCB suggests: x = {X_test[np.argmax(ucb), 0]:.3f}")
print(f"PI suggests:  x = {X_test[np.argmax(pi), 0]:.3f}")

# Blog claims:
# Current best: f(2.3) = -1.505
# EI suggests:  x = 2.598
# UCB suggests: x = 2.779
# PI suggests:  x = 2.397
