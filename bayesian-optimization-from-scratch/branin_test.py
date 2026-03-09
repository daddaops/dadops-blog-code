"""Branin Test Function — BO on a classic 2D benchmark.

Code Block 6 from the blog post.
"""
import numpy as np
from gp_bo_core import gp_posterior, expected_improvement


def branin(x1, x2):
    """Classic 2D benchmark with 3 global minima at f = 0.398."""
    a, b, c = 1, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s


# BO with 25 evaluations on [-5, 10] x [0, 15]
np.random.seed(42)
X = np.column_stack([np.random.uniform(-5, 10, 5),
                      np.random.uniform(0, 15, 5)])
y = np.array([branin(x[0], x[1]) for x in X])

x1_c, x2_c = np.linspace(-5, 10, 30), np.linspace(0, 15, 30)
XX, YY = np.meshgrid(x1_c, x2_c)
X_cand = np.column_stack([XX.ravel(), YY.ravel()])

for _ in range(20):
    mu, var = gp_posterior(X, y, X_cand,
                            length_scale=2.0, signal_var=100.0)
    ei = expected_improvement(mu, np.sqrt(var), np.min(y))
    X = np.vstack([X, X_cand[np.argmax(ei)]])
    y = np.append(y, branin(*X_cand[np.argmax(ei)]))

best = np.argmin(y)
print(f"BO found: f({X[best,0]:.2f}, {X[best,1]:.2f}) = {y[best]:.4f}")
print(f"True minimum: f = 0.3979 (at 3 locations)")
print(f"Gap: {y[best] - 0.3979:.4f}")

# Blog claims:
# BO found: f(9.48, 2.59) = 0.4179
# True minimum: f = 0.3979 (at 3 locations)
# Gap: 0.0200
