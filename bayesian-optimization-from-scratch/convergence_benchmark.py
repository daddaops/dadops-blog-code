"""Convergence Benchmark — BO vs Random on the Six-Hump Camel function.

Code Block 5 from the blog post.
"""
import numpy as np
from gp_bo_core import gp_posterior, expected_improvement


def sixhump_camel(x1, x2):
    """Classic test function: 6 local minima, global min = -1.0316."""
    return ((4 - 2.1*x1**2 + x1**4/3) * x1**2
            + x1*x2 + (-4 + 4*x2**2) * x2**2)


n_trials, n_evals, n_init = 10, 25, 5
bo_results, rand_results = [], []

for trial in range(n_trials):
    # Random search baseline
    rng = np.random.RandomState(trial)
    rand_best = min(sixhump_camel(rng.uniform(-2, 2), rng.uniform(-1, 1))
                    for _ in range(n_evals))
    rand_results.append(rand_best)

    # Bayesian Optimization
    rng2 = np.random.RandomState(trial + 100)
    X = np.column_stack([rng2.uniform(-2, 2, n_init),
                          rng2.uniform(-1, 1, n_init)])
    y = np.array([sixhump_camel(x[0], x[1]) for x in X])

    x1_c, x2_c = np.linspace(-2, 2, 25), np.linspace(-1, 1, 25)
    XX, YY = np.meshgrid(x1_c, x2_c)
    X_cand = np.column_stack([XX.ravel(), YY.ravel()])

    for _ in range(n_evals - n_init):
        mu, var = gp_posterior(X, y, X_cand,
                                length_scale=0.5, signal_var=2.0)
        ei = expected_improvement(mu, np.sqrt(var), np.min(y))
        X = np.vstack([X, X_cand[np.argmax(ei)]])
        y = np.append(y, sixhump_camel(*X_cand[np.argmax(ei)]))

    bo_results.append(np.min(y))

bo_arr, rand_arr = np.array(bo_results), np.array(rand_results)
print(f"After {n_evals} evaluations ({n_trials} trials):")
print(f"  BO:     {bo_arr.mean():.4f} +/- {bo_arr.std():.4f}")
print(f"  Random: {rand_arr.mean():.4f} +/- {rand_arr.std():.4f}")
print(f"  True minimum: -1.0316")

# Blog claims:
#   BO:     -0.9906 +/- 0.0032
#   Random: -0.7860 +/- 0.2015
