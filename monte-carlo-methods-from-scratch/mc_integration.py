"""Monte Carlo integration in 1D, 2D, and 10D.

Demonstrates that MC integration scales gracefully to high dimensions
where grid-based methods become intractable.
"""
import numpy as np

np.random.seed(42)

def mc_integrate(f, a, b, n_samples):
    """Monte Carlo integration of f over [a, b]."""
    x = np.random.uniform(a, b, n_samples)
    return (b - a) * np.mean(f(x))

# 1D: integral of e^(-x^2) from 0 to 1
estimate_1d = mc_integrate(lambda x: np.exp(-x**2), 0, 1, 100_000)
true_1d = 0.7468  # sqrt(pi)/2 * erf(1)
print(f"1D integral: {estimate_1d:.4f} (true: {true_1d:.4f})")

# 2D: integral of sin(x)*cos(y) over [0, pi] x [0, pi]
n = 100_000
xy = np.random.uniform(0, np.pi, (n, 2))
estimate_2d = (np.pi ** 2) * np.mean(np.sin(xy[:, 0]) * np.cos(xy[:, 1]))
print(f"2D integral: {estimate_2d:.4f} (true: 0.0000)")

# 10D: product of sin(xi) over [0, pi]^10
samples = np.random.uniform(0, np.pi, (500_000, 10))
integrand = np.prod(np.sin(samples), axis=1)
estimate_10d = (np.pi ** 10) * np.mean(integrand)
true_10d = 2**10  # each integral of sin(x) over [0, pi] = 2
print(f"10D integral: {estimate_10d:.1f} (true: {true_10d})")
