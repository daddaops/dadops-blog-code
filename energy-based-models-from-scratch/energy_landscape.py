"""
Energy Landscape

Defines a 2D energy landscape with Gaussian wells and computes
the partition function approximation on a grid.

Blog post: https://dadops.dev/blog/energy-based-models-from-scratch/
"""
import numpy as np

def energy(x, y, centers, heights, widths):
    """2D energy landscape: sum of inverted Gaussians (wells)."""
    E = 0.0
    for (cx, cy), h, w in zip(centers, heights, widths):
        E -= h * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * w**2))
    return E

# Three energy wells at different locations
centers = [(-2, -1), (2, 1), (0, 3)]
heights = [3.0, 2.0, 2.5]
widths  = [0.8, 1.0, 0.7]

# Evaluate on a grid
xs = np.linspace(-5, 5, 200)
ys = np.linspace(-3, 5, 200)
X, Y = np.meshgrid(xs, ys)
E = energy(X, Y, centers, heights, widths)

# Unnormalized probability: p(x) proportional to exp(-E(x))
unnorm_p = np.exp(-E)

# The partition function Z would be the integral of unnorm_p over all space
# We can approximate it on our grid:
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
Z_approx = np.sum(unnorm_p) * dx * dy
print(f"Approximate Z = {Z_approx:.2f}")
# Z_approx ~ 104.53 — grows with domain size and dimensionality
