"""GP Posterior Demo — demonstrates uncertainty near vs far from data.

Code Block 1 from the blog post.
"""
import numpy as np
from gp_bo_core import gp_posterior

# Demo: 5 noisy observations of a wiggly function
np.random.seed(42)
X_train = np.array([[-0.5], [0.2], [0.8], [1.5], [2.3]])
y_train = (np.sin(2 * X_train.ravel())
           + 0.5 * np.cos(4 * X_train.ravel())
           + 0.1 * np.random.randn(5))

X_test = np.linspace(-1, 3, 200).reshape(-1, 1)
mu, var = gp_posterior(X_train, y_train, X_test, length_scale=0.5)
sigma = np.sqrt(var)

idx_near = np.argmin(np.abs(X_test.ravel() - 0.2))  # Near training point
idx_far = -1                                          # Far from all data
print(f"Near data  (x=0.2): mu={mu[idx_near]:.3f}, sigma={sigma[idx_near]:.3f}")
print(f"Far from data (x=3): mu={mu[idx_far]:.3f}, sigma={sigma[idx_far]:.3f}")

# Blog claims:
# Near data  (x=0.2): mu=0.715, sigma=0.099  — confident!
# Far from data (x=3): mu=-0.693, sigma=0.922  — uncertain!
