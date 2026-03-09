"""The Complete BO Loop — optimizing a 1D function step by step.

Code Block 3 from the blog post.
"""
import numpy as np
from gp_bo_core import gp_posterior, expected_improvement


def objective(x):
    """A wavy function with one global minimum."""
    return np.sin(3 * x) + x**2 - 0.7 * x


# Initialize with 3 random evaluations
np.random.seed(42)
X = np.random.uniform(-2, 3, 3).reshape(-1, 1)
y = objective(X.ravel())
X_candidates = np.linspace(-2, 3, 500).reshape(-1, 1)

print("Bayesian Optimization: minimizing sin(3x) + x^2 - 0.7x")
print(f"Initial best: f({X[np.argmin(y), 0]:.3f}) = {np.min(y):.4f}\n")

for i in range(12):
    mu, var = gp_posterior(X, y, X_candidates, length_scale=0.5)
    sigma = np.sqrt(var)
    ei = expected_improvement(mu, sigma, np.min(y))

    next_idx = np.argmax(ei)
    next_x = X_candidates[next_idx, 0]
    next_y = objective(next_x)

    X = np.vstack([X, [[next_x]]])
    y = np.append(y, next_y)
    best_idx = np.argmin(y)
    print(f"Iter {i+1:2d}: eval x={next_x:7.3f}, f(x)={next_y:7.4f}  "
          f"| best: f({X[best_idx,0]:.3f})={y[best_idx]:.4f}")

# Blog claims:
# Initial best: f(-0.127) = -0.2674
# Iter  4: f(-0.327) = -0.4951 (breakthrough)
# Iter  7: f(-0.357) = -0.5003 (converged)
# Final: gap ~0.0001 from true optimum -0.5004
