"""Vanilla SGD on a simple bowl loss surface.

L(x, y) = x² + y² — smooth, isotropic, easy to optimize.
"""
import numpy as np

class SGD:
    """Vanilla stochastic gradient descent."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """Update parameters using raw gradients."""
        return [p - self.lr * g for p, g in zip(params, grads)]

# A simple bowl: L(x, y) = x² + y²
# Gradient: [2x, 2y]
def bowl_loss(pos):
    return pos[0]**2 + pos[1]**2

def bowl_grad(pos):
    return np.array([2*pos[0], 2*pos[1]])

opt = SGD(lr=0.1)
pos = np.array([4.0, 3.0])

for step in range(15):
    loss = bowl_loss(pos)
    grad = bowl_grad(pos)
    print(f"Step {step:2d}: pos=({pos[0]:6.3f}, {pos[1]:6.3f})  loss={loss:.4f}")
    pos = np.array(opt.step([pos[0], pos[1]], [grad[0], grad[1]]))

# Step  0: pos=( 4.000,  3.000)  loss=25.0000
# Step  1: pos=( 3.200,  2.400)  loss=16.0000
# Step  2: pos=( 2.560,  1.920)  loss=10.2400
# ...
# Step 14: pos=( 0.176,  0.132)  loss=0.0484
