"""
E-Step and M-Step Implementation

Implements the two core steps of EM for Gaussian Mixture Models:
- E-step: compute responsibilities (soft assignments)
- M-step: update parameters using responsibility-weighted data

Blog post: https://dadops.dev/blog/expectation-maximization-from-scratch/
"""
import numpy as np
from gmm_data import make_gmm_data


def multivariate_gaussian(X, mu, cov):
    """Evaluate the multivariate Gaussian density at each row of X."""
    d = X.shape[1]
    diff = X - mu
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return norm * np.exp(exponent)


def e_step(X, weights, means, covs):
    """Compute responsibilities: gamma[i, k] = p(z_i = k | x_i, params)."""
    N, K = len(X), len(weights)
    gamma = np.zeros((N, K))
    for k in range(K):
        gamma[:, k] = weights[k] * multivariate_gaussian(X, means[k], covs[k])
    # Normalize so each row sums to 1
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def m_step(X, gamma):
    """Update GMM parameters using responsibilities as weights."""
    N, d = X.shape
    K = gamma.shape[1]

    # Effective number of points per component
    N_k = gamma.sum(axis=0)                      # shape (K,)

    # Update mixing weights
    weights = N_k / N                             # shape (K,)

    # Update means (responsibility-weighted centroids)
    means = []
    for k in range(K):
        mu_k = (gamma[:, k:k+1] * X).sum(axis=0) / N_k[k]
        means.append(mu_k)

    # Update covariances (using the NEW means)
    covs = []
    for k in range(K):
        diff = X - means[k]                       # (N, d)
        weighted_diff = gamma[:, k:k+1] * diff    # (N, d)
        cov_k = (weighted_diff.T @ diff) / N_k[k] # (d, d)
        # Add small regularization to prevent singular covariances
        cov_k += 1e-6 * np.eye(d)
        covs.append(cov_k)

    return list(weights), means, covs


if __name__ == "__main__":
    X, z_true, true_w, true_mu, true_cov = make_gmm_data()

    # E-step: compute responsibilities with initial guesses
    weights_init = [1/3, 1/3, 1/3]
    means_init = [np.array([-1, -1]), np.array([4, 0]), np.array([1, 3])]
    covs_init = [np.eye(2)] * 3

    gamma = e_step(X, weights_init, means_init, covs_init)
    # Show responsibilities for a few points
    for i in [0, 100, 300]:
        print(f"Point {i}: responsibilities = [{gamma[i,0]:.3f}, "
              f"{gamma[i,1]:.3f}, {gamma[i,2]:.3f}]")

    print()

    # M-step: update parameters using responsibilities
    weights_new, means_new, covs_new = m_step(X, gamma)
    print("Updated mixing weights:", [f"{w:.3f}" for w in weights_new])
    print("Updated means:")
    for k, mu in enumerate(means_new):
        print(f"  Component {k}: [{mu[0]:.2f}, {mu[1]:.2f}]")
