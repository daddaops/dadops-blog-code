"""Head-to-head optimizer comparison on a Beale-like surface.

Compares SGD, Momentum, RMSProp, and Adam on a challenging
surface with a ravine and curvature. Minimum near (3, 0.5).
"""
import numpy as np

class SGD:
    """Vanilla stochastic gradient descent."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]

class SGDMomentum:
    """SGD with momentum."""
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(g) for g in grads]
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocity[i] = self.beta * self.velocity[i] + g
            updated.append(p - self.lr * self.velocity[i])
        return updated

class RMSProp:
    """RMSProp — per-parameter adaptive learning rates."""
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sq_avg = None

    def step(self, params, grads):
        if self.sq_avg is None:
            self.sq_avg = [np.zeros_like(g) for g in grads]
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.sq_avg[i] = self.beta * self.sq_avg[i] + (1 - self.beta) * g**2
            updated.append(p - self.lr * g / (np.sqrt(self.sq_avg[i]) + self.eps))
        return updated

class Adam:
    """Adam — adaptive moment estimation with bias correction."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(g) for g in grads]
            self.v = [np.zeros_like(g) for g in grads]
        self.t += 1
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            updated.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return updated

# Beale-like surface: ravine + curvature
# Minimum near (3, 0.5)
def loss_fn(pos):
    x, y = pos
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2

def grad_fn(pos):
    x, y = pos
    a = 1.5 - x + x*y
    b = 2.25 - x + x*y**2
    dldx = 2*a*(-1 + y) + 2*b*(-1 + y**2)
    dldy = 2*a*x + 2*b*(2*x*y)
    return np.array([dldx, dldy])

start = np.array([0.5, 3.5])
optimizers = {
    "SGD":      SGD(lr=0.0005),
    "Momentum": SGDMomentum(lr=0.0005, beta=0.9),
    "RMSProp":  RMSProp(lr=0.005, beta=0.9),
    "Adam":     Adam(lr=0.01, beta1=0.9, beta2=0.999),
}

results = {}
for name, opt in optimizers.items():
    pos = start.copy()
    trajectory = [pos.copy()]
    for step in range(200):
        grad = grad_fn(pos)
        grad = np.clip(grad, -10, 10)  # clip for stability
        pos = np.array(opt.step([pos[0], pos[1]], [grad[0], grad[1]]))
        trajectory.append(pos.copy())
    results[name] = {"final_loss": loss_fn(pos), "final_pos": pos, "steps": trajectory}

print(f"{'Optimizer':<12} {'Final Loss':>12}  {'Final Position':>20}")
print("-" * 48)
for name, r in results.items():
    pos = r['final_pos']
    print(f"{name:<12} {r['final_loss']:12.6f}  ({pos[0]:.4f}, {pos[1]:.4f})")

# Optimizer     Final Loss       Final Position
# ------------------------------------------------
# SGD              0.864977  (-0.2776, 3.1417)
# Momentum         0.652723  (-0.5086, 2.4178)
# RMSProp          0.852514  (-0.2874, 3.0942)
# Adam             0.821062  (-0.3134, 2.9774)
