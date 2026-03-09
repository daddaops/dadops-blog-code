"""Gradient explosion in deep networks without normalization.

Shows how the standard deviation of activations grows
exponentially across 50 layers.
"""
import numpy as np

np.random.seed(42)
d = 128
x = np.random.randn(4, d)

print(f"Input:     std(x) = {x.std():.4f}")
for i in range(50):
    W = np.random.randn(d, d) * 0.15
    x = x @ W
    if i in [0, 9, 19, 29, 49]:
        print(f"Layer {i+1:2d}:  std(x) = {x.std():.6e}")
