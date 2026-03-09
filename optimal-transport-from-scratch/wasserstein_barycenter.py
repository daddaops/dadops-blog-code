"""Wasserstein barycenter of 1D histograms.

Uses iterative Bregman projections (Cuturi & Doucet 2014)
to find the distribution that minimizes average Wasserstein distance
to a set of input distributions.
"""
import numpy as np

def wasserstein_barycenter(histograms, support, weights=None,
                           epsilon=0.05, max_iter=50):
    """Compute Wasserstein barycenter of 1D histograms.
    Uses iterative Bregman projections (Cuturi & Doucet 2014).
    histograms: list of K histograms (each sums to 1)
    support: shared bin centers
    weights: barycentric weights (uniform if None)"""
    K_dists = len(histograms)
    n_bins = len(support)
    if weights is None:
        weights = np.ones(K_dists) / K_dists

    # Cost matrix (normalized to prevent kernel underflow)
    C = (support[:, None] - support[None, :]) ** 2
    C /= C.max()
    K = np.exp(-C / epsilon)  # Gibbs kernel

    # Initialize barycenter as uniform
    bary = np.ones(n_bins) / n_bins

    for iteration in range(max_iter):
        log_bary = np.zeros(n_bins)
        for k in range(K_dists):
            # Sinkhorn scaling for transport from bary to histogram k
            u = np.ones(n_bins)
            for _ in range(100):
                v = histograms[k] / (K.T @ u)
                u = bary / (K @ v)
            # K @ v transports histogram k into barycenter's frame
            log_bary += weights[k] * np.log(np.maximum(K @ v, 1e-16))
        bary = np.exp(log_bary)
        bary /= bary.sum()

    return bary

# Three 1D distributions: left peak, center peak, right peak
bins = np.linspace(0, 10, 51)
h1 = np.exp(-0.5 * ((bins - 2) / 0.8) ** 2)
h2 = np.exp(-0.5 * ((bins - 5) / 1.0) ** 2)
h3 = np.exp(-0.5 * ((bins - 8) / 0.6) ** 2)
for h in [h1, h2, h3]:
    h /= h.sum()

bary = wasserstein_barycenter([h1, h2, h3], bins, epsilon=0.05)
peak_pos = bins[np.argmax(bary)]
print(f"Barycenter peak at x={peak_pos:.1f} (mean of 2, 5, 8 = 5.0)")
# Barycenter peak at x=5.0 (mean of 2, 5, 8 = 5.0)
