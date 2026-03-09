"""Minimal Neural ODE: dh/dt = W2 @ tanh(W1 @ h + b1) + b2.

The forward pass IS solving an ODE — no discrete layers.
Uses RK4 integration with configurable step count.
"""
import numpy as np


class NeuralODE:
    """A tiny Neural ODE: dh/dt = W2 @ tanh(W1 @ h + b1) + b2."""
    def __init__(self, dim, hidden=16):
        scale = np.sqrt(2 / dim)
        self.W1 = np.random.randn(hidden, dim) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(dim, hidden) * scale
        self.b2 = np.zeros(dim)

    def f(self, h, t):
        """The learned dynamics f_theta(h, t)."""
        z = np.tanh(self.W1 @ h + self.b1)
        return self.W2 @ z + self.b2

    def forward(self, h0, t0=0.0, t1=1.0, steps=20):
        """Solve the IVP from t0 to t1 using RK4."""
        dt = (t1 - t0) / steps
        h = h0.copy()
        trajectory = [h.copy()]
        for i in range(steps):
            t = t0 + i * dt
            k1 = self.f(h, t)
            k2 = self.f(h + 0.5*dt*k1, t + 0.5*dt)
            k3 = self.f(h + 0.5*dt*k2, t + 0.5*dt)
            k4 = self.f(h + dt*k3, t + dt)
            h = h + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory.append(h.copy())
        return h, trajectory


if __name__ == "__main__":
    np.random.seed(42)
    node = NeuralODE(dim=2)
    h_final, path = node.forward(np.array([1.0, 0.5]))
    print(f"Input: [1.0, 0.5] -> Output: [{h_final[0]:.3f}, {h_final[1]:.3f}]")
    print(f"Trajectory length: {len(path)} points")
    print(f"Parameters: W1={node.W1.shape}, W2={node.W2.shape}")
