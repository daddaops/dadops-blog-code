"""
GMM Data Generation

Generates synthetic 2D data from a known 3-component Gaussian Mixture Model
with different covariance structures per component.

Blog post: https://dadops.dev/blog/expectation-maximization-from-scratch/
"""
import numpy as np


def make_gmm_data(seed=42):
    """Generate 2D data from a known 3-component GMM."""
    rng = np.random.RandomState(seed)
    n_samples = 500

    # True parameters (what EM will try to recover)
    weights = [0.35, 0.40, 0.25]
    means = [np.array([0, 0]),
             np.array([5, 0]),
             np.array([2, 4])]
    covs = [np.array([[0.8, 0.2], [0.2, 0.6]]),         # slightly tilted
            np.array([[2.5, 0.0], [0.0, 0.15]]),         # wide and flat
            np.array([[0.6, -0.5], [-0.5, 1.5]])]        # tilted other way

    X, z_true = [], []
    for i in range(n_samples):
        k = rng.choice(3, p=weights)
        x = rng.multivariate_normal(means[k], covs[k])
        X.append(x)
        z_true.append(k)
    return np.array(X), np.array(z_true), weights, means, covs


if __name__ == "__main__":
    X, z_true, true_w, true_mu, true_cov = make_gmm_data()
    print(f"Generated {len(X)} points from 3-component GMM")
    print(f"Component sizes: {[np.sum(z_true==k) for k in range(3)]}")
