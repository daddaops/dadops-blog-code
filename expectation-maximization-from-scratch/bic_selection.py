"""
BIC-Based Model Selection

Uses the Bayesian Information Criterion to select the optimal number
of mixture components, balancing fit against complexity.

Blog post: https://dadops.dev/blog/expectation-maximization-from-scratch/
"""
import numpy as np
from gmm_data import make_gmm_data
from em_full import fit_gmm


def compute_bic(X, K, seed=0):
    """Fit a GMM with K components and return BIC."""
    w, mu, cov, _, lls = fit_gmm(X, K, max_iter=200, seed=seed)
    N, d = X.shape
    n_params = K * d + K * d * (d + 1) // 2 + (K - 1)
    log_likelihood = lls[-1]
    bic = -2 * log_likelihood + n_params * np.log(N)
    return bic, log_likelihood, n_params


if __name__ == "__main__":
    X, z_true, true_w, true_mu, true_cov = make_gmm_data()

    print(f"{'K':>2} | {'Params':>6} | {'Log-L':>10} | {'BIC':>10}")
    print("-" * 40)
    bics = []
    for K in range(1, 7):
        bic, ll, p = compute_bic(X, K, seed=7)
        bics.append(bic)
        marker = " <-- best" if K == np.argmin(bics) + 1 and K > 1 else ""
        print(f"{K:2d} | {p:6d} | {ll:10.1f} | {bic:10.1f}{marker}")

    best_K = np.argmin(bics) + 1
    print(f"\nBIC selects K={best_K} components (true K=3)")
