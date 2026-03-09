"""Estimating π with random darts.

Throw random points in a unit square, count those inside the
quarter circle — the ratio converges to π/4.
"""
import numpy as np

np.random.seed(42)

for n in [100, 1_000, 10_000, 100_000]:
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    inside = np.sum(x**2 + y**2 <= 1.0)
    pi_hat = 4.0 * inside / n
    error = abs(pi_hat - np.pi)
    print(f"N = {n:,}  π ≈ {pi_hat:.6f}  |error| = {error:.6f}")
