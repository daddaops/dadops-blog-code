"""Augmented Neural ODE: lift to higher dimensions to break topology.

Standard 2D Neural ODE preserves topology (homeomorphism).
Augmenting with extra dimensions allows trajectories to detour
through higher-dimensional space where they CAN separate.
"""
import numpy as np


class AugmentedNeuralODE:
    """Lift input to higher-dim space, then solve ODE."""
    def __init__(self, data_dim, aug_dim=3, hidden=32):
        total = data_dim + aug_dim
        self.aug_dim = aug_dim
        scale = np.sqrt(2 / total)
        self.W1 = np.random.randn(hidden, total) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(total, hidden) * scale
        self.b2 = np.zeros(total)

    def f(self, h, t):
        z = np.tanh(self.W1 @ h + self.b1)
        return self.W2 @ z + self.b2

    def forward(self, x, t1=1.0, steps=20):
        # Augment: pad input with zeros
        h = np.concatenate([x, np.zeros(self.aug_dim)])
        dt = t1 / steps
        for _ in range(steps):
            k1 = self.f(h, 0)
            k2 = self.f(h + 0.5*dt*k1, 0)
            k3 = self.f(h + 0.5*dt*k2, 0)
            k4 = self.f(h + dt*k3, 0)
            h = h + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return h[:len(x)]  # project back to data dimension


if __name__ == "__main__":
    np.random.seed(42)

    # Standard NODE (2D -> 2D)
    from neural_ode import NeuralODE
    std_node = NeuralODE(dim=2, hidden=16)
    h_std, _ = std_node.forward(np.array([1.0, 0.0]))
    print(f"Standard NODE: [1.0, 0.0] -> [{h_std[0]:.3f}, {h_std[1]:.3f}]")

    # Augmented NODE (2D + 3 extra dims = 5D total)
    aug_node = AugmentedNeuralODE(data_dim=2, aug_dim=3, hidden=32)
    h_aug = aug_node.forward(np.array([1.0, 0.0]))
    print(f"Augmented NODE: [1.0, 0.0] -> [{h_aug[0]:.3f}, {h_aug[1]:.3f}]")
    print(f"  (trajectories detour through {2 + 3}D space)")

    # Show that different inputs produce different outputs
    for x in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]:
        out = aug_node.forward(np.array(x))
        print(f"  {x} -> [{out[0]:.3f}, {out[1]:.3f}]")
