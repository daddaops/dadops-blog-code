"""Wasserstein distance vs KL divergence comparison.

Shows how W1, KL, and JS divergence behave as two Gaussians drift apart.
W1 scales linearly with shift; KL grows quadratically; JS saturates.
"""
import numpy as np

def wasserstein_1d(samples_p, samples_q):
    """W1 distance between 1D samples via CDF area."""
    all_pts = np.sort(np.unique(np.concatenate([samples_p, samples_q])))
    # Empirical CDFs at each point
    cdf_p = np.searchsorted(np.sort(samples_p), all_pts, side='right') / len(samples_p)
    cdf_q = np.searchsorted(np.sort(samples_q), all_pts, side='right') / len(samples_q)
    # Trapezoidal integration of |F - G|
    diffs = np.abs(cdf_p - cdf_q)
    dx = np.diff(all_pts, prepend=all_pts[0] - 0.5, append=all_pts[-1] + 0.5)
    return np.sum(diffs * dx[1:])

def kl_divergence(p_hist, q_hist):
    """KL(P || Q) from histograms. Returns inf if Q=0 where P>0."""
    mask = p_hist > 0
    if np.any((q_hist[mask] == 0)):
        return float('inf')
    return np.sum(p_hist[mask] * np.log(p_hist[mask] / q_hist[mask]))

def js_divergence(p_hist, q_hist):
    """Jensen-Shannon divergence from histograms."""
    m = 0.5 * (p_hist + q_hist)
    return 0.5 * kl_divergence(p_hist, m) + 0.5 * kl_divergence(q_hist, m)

# Compare as two Gaussians drift apart
np.random.seed(7)
n = 5000
bins = np.linspace(-8, 16, 200)

print(f"{'Shift':>6}  {'W1':>8}  {'KL':>10}  {'JS':>8}")
print("-" * 38)
for shift in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    p = np.random.normal(0, 1, n)
    q = np.random.normal(shift, 1, n)
    w1 = wasserstein_1d(p, q)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist / p_hist.sum() + 1e-10
    q_hist = q_hist / q_hist.sum() + 1e-10
    kl = kl_divergence(p_hist, q_hist)
    js = js_divergence(p_hist, q_hist)
    print(f"{shift:6.1f}  {w1:8.3f}  {kl:10.3f}  {js:8.4f}")
