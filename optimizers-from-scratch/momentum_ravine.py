"""SGD with Momentum on a ravine loss surface.

Momentum accumulates velocity, letting the optimizer build speed
along the ravine floor instead of oscillating across it.
"""
import numpy as np

class SGDMomentum:
    """SGD with momentum — accumulates velocity from past gradients."""
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(g) for g in grads]

        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            # Accumulate velocity: keep β of old direction, add new gradient
            self.velocity[i] = self.beta * self.velocity[i] + g
            # Step in the velocity direction
            updated.append(p - self.lr * self.velocity[i])
        return updated

def ravine_loss(pos):
    return 50 * pos[0]**2 + pos[1]**2

def ravine_grad(pos):
    return np.array([100 * pos[0], 2 * pos[1]])

opt = SGDMomentum(lr=0.01, beta=0.9)
pos = np.array([1.0, 8.0])

for step in range(30):
    loss = ravine_loss(pos)
    grad = ravine_grad(pos)
    if step % 5 == 0:
        print(f"Step {step:2d}: pos=({pos[0]:7.4f}, {pos[1]:7.4f})  loss={loss:.4f}")
    pos = np.array(opt.step([pos[0], pos[1]], [grad[0], grad[1]]))

# Step  0: pos=( 1.0000,  8.0000)  loss=114.0000
# Step  5: pos=( 0.8019,  5.9945)  loss=68.0866
# Step 10: pos=( 0.1753,  2.4570)  loss=7.5734
# Step 15: pos=(-0.3061, -0.5965)  loss=5.0403
# Step 20: pos=(-0.3958, -2.1654)  loss=12.5231
# Step 25: pos=(-0.1973, -2.2417)  loss=6.9710
