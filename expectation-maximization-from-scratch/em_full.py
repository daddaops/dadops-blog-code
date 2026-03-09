"""
Complete EM Algorithm

Full expectation-maximization for Gaussian Mixture Models with
convergence tracking and monotonic log-likelihood guarantee.

Blog post: https://dadops.dev/blog/expectation-maximization-from-scratch/
"""
import numpy as np
from gmm_data import make_gmm_data


def fit_gmm(X, K, max_iter=100, tol=1e-6, seed=0):
    """Fit a K-component GMM using the EM algorithm."""
    rng = np.random.RandomState(seed)
    N, d = X.shape

    # Initialize: random means from data, identity covariances, uniform weights
    indices = rng.choice(N, K, replace=False)
    means = [X[idx].copy() for idx in indices]
    covs = [np.eye(d) * np.var(X, axis=0).mean() for _ in range(K)]
    weights = [1.0 / K] * K

    log_likelihoods = []

    for iteration in range(max_iter):
        # --- E-step ---
        gamma = np.zeros((N, K))
        for k in range(K):
            diff = X - means[k]
            cov_inv = np.linalg.inv(covs[k])
            det = np.linalg.det(covs[k])
            norm = 1.0 / np.sqrt((2 * np.pi) ** d * det)
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            gamma[:, k] = weights[k] * norm * np.exp(exponent)
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma /= gamma_sum

        # --- Log-likelihood ---
        ll = np.sum(np.log(gamma_sum.ravel()))
        log_likelihoods.append(ll)
        if len(log_likelihoods) > 1 and ll - log_likelihoods[-2] < tol:
            break

        # --- M-step ---
        N_k = gamma.sum(axis=0)
        weights = list(N_k / N)
        new_means = []
        new_covs = []
        for k in range(K):
            mu_k = (gamma[:, k:k+1] * X).sum(axis=0) / N_k[k]
            new_means.append(mu_k)
            diff = X - mu_k
            cov_k = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
            cov_k += 1e-6 * np.eye(d)
            new_covs.append(cov_k)
        means = new_means
        covs = new_covs

    return weights, means, covs, gamma, log_likelihoods


if __name__ == "__main__":
    X, z_true, true_w, true_mu, true_cov = make_gmm_data()

    # Fit a 3-component GMM
    w, mu, cov, gamma, lls = fit_gmm(X, K=3, seed=7)
    print(f"Converged in {len(lls)} iterations")
    print(f"Log-likelihood: {lls[0]:.1f} -> {lls[-1]:.1f} (monotonic increase)")
    print(f"Recovered weights: {[f'{v:.3f}' for v in w]}")
    for k in range(3):
        print(f"Component {k}: mean=[{mu[k][0]:.2f}, {mu[k][1]:.2f}]")

    # Verify monotonic increase
    diffs = [lls[i+1] - lls[i] for i in range(len(lls)-1)]
    print(f"All LL increases non-negative: {all(d >= -1e-10 for d in diffs)}")
