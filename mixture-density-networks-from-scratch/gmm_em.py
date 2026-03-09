"""Fit a 2-component GMM via the EM algorithm.

Demonstrates the Expectation-Maximization algorithm on bimodal data
as a building block for understanding MDN training.
"""
import numpy as np

# Generate bimodal data: two clusters
np.random.seed(7)
data = np.concatenate([
    np.random.normal(-2.0, 0.5, 300),
    np.random.normal(2.0, 0.8, 200)
])

# Fit a 2-component GMM via EM (simplified)
K = 2
mu = np.array([-1.0, 1.0])      # initial means
sigma = np.array([1.0, 1.0])     # initial std devs
pi = np.array([0.5, 0.5])        # initial mixing weights

for step in range(50):
    # E-step: compute responsibilities
    resp = np.zeros((len(data), K))
    for k in range(K):
        resp[:, k] = pi[k] * np.exp(-0.5 * ((data - mu[k]) / sigma[k])**2) / (sigma[k] * np.sqrt(2 * np.pi))
    resp /= resp.sum(axis=1, keepdims=True)

    # M-step: update parameters
    for k in range(K):
        Nk = resp[:, k].sum()
        mu[k] = (resp[:, k] * data).sum() / Nk
        sigma[k] = np.sqrt((resp[:, k] * (data - mu[k])**2).sum() / Nk)
        pi[k] = Nk / len(data)

print(f"Component 1: mu={mu[0]:.2f}, sigma={sigma[0]:.2f}, pi={pi[0]:.2f}")
print(f"Component 2: mu={mu[1]:.2f}, sigma={sigma[1]:.2f}, pi={pi[1]:.2f}")
# Output: Component 1: mu=-2.01, sigma=0.50, pi=0.60
#         Component 2: mu=1.98, sigma=0.81, pi=0.40
