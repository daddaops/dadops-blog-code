"""Vanilla SGD on a ravine loss surface.

L(x, y) = 50x² + y² — steep across x, gentle along y.
SGD oscillates across the ravine and crawls along it.
"""
import numpy as np

class SGD:
    """Vanilla stochastic gradient descent."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """Update parameters using raw gradients."""
        return [p - self.lr * g for p, g in zip(params, grads)]

# A ravine: L(x, y) = 50x² + y²
# Steep across x, gentle along y
# Gradient: [100x, 2y]
def ravine_loss(pos):
    return 50 * pos[0]**2 + pos[1]**2

def ravine_grad(pos):
    return np.array([100 * pos[0], 2 * pos[1]])

opt = SGD(lr=0.01)  # can't use 0.1 — would diverge!
pos = np.array([1.0, 8.0])

for step in range(30):
    loss = ravine_loss(pos)
    grad = ravine_grad(pos)
    if step % 5 == 0:
        print(f"Step {step:2d}: pos=({pos[0]:7.4f}, {pos[1]:7.4f})  loss={loss:.4f}")
    pos = np.array(opt.step([pos[0], pos[1]], [grad[0], grad[1]]))

# Step  0: pos=( 1.0000,  8.0000)  loss=114.0000
# Step  5: pos=( 0.0000,  7.2314)  loss=52.2927
# Step 10: pos=( 0.0000,  6.5366)  loss=42.7269
# Step 15: pos=( 0.0000,  5.9086)  loss=34.9110
# Step 20: pos=( 0.0000,  5.3409)  loss=28.5248
# Step 25: pos=( 0.0000,  4.8277)  loss=23.3069
