"""MDN loss function: negative log-likelihood with log-sum-exp trick.

Computes the NLL of observed targets under a predicted Gaussian mixture,
using the log-sum-exp trick for numerical stability.
"""
import numpy as np

def mdn_loss(pi, mu, sigma, y):
    """Negative log-likelihood of mixture density, numerically stable."""
    K = pi.shape[1]
    y = y.reshape(-1, 1)  # (N, 1)

    # Log of each Gaussian component: log N(y | mu_k, sigma_k^2)
    # Note: np.pi here is the math constant 3.14159..., not the mixing weights
    log_gauss = -0.5 * np.log(2 * np.pi) - np.log(sigma) \
                - 0.5 * ((y - mu) / sigma) ** 2             # (N, K)

    # Log mixing coefficients
    log_pi = np.log(pi + 1e-10)                              # (N, K)

    # Log-sum-exp for numerical stability
    log_components = log_pi + log_gauss                       # (N, K)
    max_log = log_components.max(axis=1, keepdims=True)       # (N, 1)
    log_sum = max_log + np.log(
        np.exp(log_components - max_log).sum(axis=1, keepdims=True)
    )

    nll = -log_sum.mean()
    return nll

# Example: compute loss on random MDN output vs target data
np.random.seed(42)
N = 100
pi_ex = np.full((N, 3), 1/3)                   # uniform mixing
mu_ex = np.column_stack([
    np.full(N, -1.0), np.full(N, 0.0), np.full(N, 1.0)
])
sigma_ex = np.full((N, 3), 0.5)
y_ex = np.random.choice([-1, 0, 1], N) + np.random.normal(0, 0.3, N)

loss = mdn_loss(pi_ex, mu_ex, sigma_ex, y_ex)
print(f"MDN NLL loss: {loss:.3f}")  # ~1.24 (reasonable for this setup)
