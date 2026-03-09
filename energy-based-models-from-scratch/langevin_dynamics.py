"""
Langevin Dynamics

MCMC sampling from a 2D energy landscape using Langevin dynamics.
Samples cluster around energy minima (high probability regions).

Blog post: https://dadops.dev/blog/energy-based-models-from-scratch/
"""
import numpy as np

def energy_2d(xy, centers, heights, widths):
    """Compute energy for array of 2D points."""
    E = np.zeros(len(xy))
    for (cx, cy), h, w in zip(centers, heights, widths):
        E -= h * np.exp(-((xy[:,0]-cx)**2 + (xy[:,1]-cy)**2) / (2*w**2))
    return E

def energy_grad(xy, centers, heights, widths):
    """Gradient of the energy function (analytical)."""
    grad = np.zeros_like(xy)
    for (cx, cy), h, w in zip(centers, heights, widths):
        diff_x = xy[:,0] - cx
        diff_y = xy[:,1] - cy
        gauss = h * np.exp(-(diff_x**2 + diff_y**2) / (2*w**2))
        grad[:,0] += gauss * diff_x / w**2
        grad[:,1] += gauss * diff_y / w**2
    return grad

# Langevin dynamics: sample from the energy landscape
np.random.seed(42)
centers = [(-2, -1), (2, 1), (0, 3)]
heights = [3.0, 2.0, 2.5]
widths  = [0.8, 1.0, 0.7]

n_samples = 300
eta = 0.05  # step size
xy = np.random.randn(n_samples, 2) * 3  # start from random locations

for step in range(200):
    grad = energy_grad(xy, centers, heights, widths)
    noise = np.random.randn(n_samples, 2)
    xy = xy - eta * grad + np.sqrt(2 * eta) * noise

# After 200 steps, samples cluster around the three energy minima
print(f"Sample mean energies per cluster:")
for c in centers:
    nearby = xy[np.linalg.norm(xy - c, axis=1) < 1.5]
    if len(nearby) > 0:
        print(f"  Near {c}: {len(nearby)} samples, mean E = "
              f"{np.mean(energy_2d(nearby, centers, heights, widths)):.2f}")
