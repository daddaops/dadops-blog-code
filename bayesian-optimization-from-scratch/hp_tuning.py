"""Hyperparameter Tuning — Grid vs Random vs BO on a 2D accuracy landscape.

Code Block 4 from the blog post.
"""
import numpy as np
from gp_bo_core import gp_posterior, expected_improvement


def hp_objective(log_lr, log_l2):
    """Simulated accuracy landscape. Peak near lr=0.01, l2=0.001."""
    return 0.65 + 0.28 * np.exp(
        -((log_lr + 2)**2 / 0.5 + (log_l2 + 3)**2 / 1.0))


rng = np.random.RandomState(42)

# Strategy 1: Grid search (5x5 = 25 evaluations)
grid_best = 0
for lr in np.linspace(-4, -1, 5):
    for l2 in np.linspace(-5, -1, 5):
        acc = hp_objective(lr, l2) + 0.02 * rng.randn()
        grid_best = max(grid_best, acc)

# Strategy 2: Random search (25 evaluations)
rng2 = np.random.RandomState(123)
rand_best = 0
for _ in range(25):
    lr, l2 = rng2.uniform(-4, -1), rng2.uniform(-5, -1)
    acc = hp_objective(lr, l2) + 0.02 * rng2.randn()
    rand_best = max(rand_best, acc)

# Strategy 3: Bayesian Optimization (5 init + 20 BO = 25 total)
rng3 = np.random.RandomState(456)
X_bo = np.column_stack([rng3.uniform(-4, -1, 5),
                         rng3.uniform(-5, -1, 5)])
y_neg = np.array([-hp_objective(x[0], x[1]) + 0.02 * rng3.randn()
                   for x in X_bo])  # Negate: BO minimizes

lr_c, l2_c = np.linspace(-4, -1, 30), np.linspace(-5, -1, 30)
XX, YY = np.meshgrid(lr_c, l2_c)
X_cand = np.column_stack([XX.ravel(), YY.ravel()])

for _ in range(20):
    mu, var = gp_posterior(X_bo, y_neg, X_cand, length_scale=1.0)
    ei = expected_improvement(mu, np.sqrt(var), np.min(y_neg))
    best_cand = X_cand[np.argmax(ei)]
    X_bo = np.vstack([X_bo, best_cand])
    y_neg = np.append(y_neg,
        -hp_objective(*best_cand) + 0.02 * rng3.randn())

print(f"Grid search (25 evals):   {grid_best:.1%} accuracy")
print(f"Random search (25 evals): {rand_best:.1%} accuracy")
print(f"Bayesian opt (25 evals):  {-np.min(y_neg):.1%} accuracy")

# Blog claims:
# Grid search (25 evals):   90.3% accuracy
# Random search (25 evals): 92.0% accuracy
# Bayesian opt (25 evals):  94.6% accuracy
