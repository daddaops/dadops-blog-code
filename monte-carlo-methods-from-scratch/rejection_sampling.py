"""Rejection sampling from a mixture of Gaussians.

Uses a broad Gaussian proposal to sample from a bimodal target
distribution, demonstrating the accept/reject mechanism.
"""
import numpy as np
from scipy.stats import norm

np.random.seed(42)

# Target: mixture of two Gaussians
def target_pdf(x):
    return 0.3 * norm.pdf(x, -2, 0.7) + 0.7 * norm.pdf(x, 3, 1.2)

# Proposal: single broad Gaussian that covers both modes
def proposal_pdf(x):
    return norm.pdf(x, 1, 3)

# Find M such that M * q(x) >= p(x) everywhere
x_grid = np.linspace(-6, 8, 10000)
M = np.max(target_pdf(x_grid) / proposal_pdf(x_grid)) * 1.01

# Rejection sampling
n_proposals = 100_000
proposals = np.random.normal(1, 3, n_proposals)
u = np.random.uniform(0, 1, n_proposals)
accept_prob = target_pdf(proposals) / (M * proposal_pdf(proposals))
accepted = proposals[u <= accept_prob]

acceptance_rate = len(accepted) / n_proposals
print(f"Envelope constant M: {M:.2f}")
print(f"Proposed: {n_proposals:,}  Accepted: {len(accepted):,}")
print(f"Acceptance rate: {acceptance_rate:.1%}")
