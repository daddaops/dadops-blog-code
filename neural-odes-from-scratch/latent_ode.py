"""Latent ODE for irregular time series.

Encode -> latent dynamics -> decode. Handles any observation
spacing by solving the ODE to exactly the requested times.
"""
import numpy as np


class LatentODE:
    """Simplified Latent ODE: encode -> latent dynamics -> decode."""
    def __init__(self, obs_dim=1, latent_dim=4):
        self.latent_dim = latent_dim
        # Encoder: observations -> latent initial state
        self.enc_W = np.random.randn(latent_dim, obs_dim + 1) * 0.5
        # Dynamics: dz/dt = tanh(W_dyn @ z)
        self.dyn_W = np.random.randn(latent_dim, latent_dim) * 0.3
        # Decoder: latent state -> observation
        self.dec_W = np.random.randn(obs_dim, latent_dim) * 0.5

    def encode(self, obs_times, obs_values):
        """Simple mean encoding (real version uses ODE-RNN)."""
        features = np.column_stack([obs_times, obs_values])
        return np.tanh(self.enc_W @ features.mean(axis=0))

    def latent_dynamics(self, z0, eval_times, steps_per_unit=10):
        """Solve latent ODE at arbitrary evaluation times."""
        results = {}
        z = z0.copy()
        t_cur = 0.0
        for t_target in sorted(eval_times):
            n = max(1, int((t_target - t_cur) * steps_per_unit))
            dt = (t_target - t_cur) / n
            for _ in range(n):
                z = z + dt * np.tanh(self.dyn_W @ z)
            results[t_target] = z.copy()
            t_cur = t_target
        return results

    def decode(self, z):
        return self.dec_W @ z


if __name__ == "__main__":
    np.random.seed(42)
    model = LatentODE(obs_dim=1, latent_dim=4)

    # Irregular observations (non-uniform spacing)
    obs_times = np.array([0.0, 0.3, 0.7, 1.5, 2.8])
    obs_values = np.sin(obs_times).reshape(-1, 1)

    # Encode observations to latent state
    z0 = model.encode(obs_times.reshape(-1, 1), obs_values)
    print(f"Latent initial state: [{', '.join(f'{z:.3f}' for z in z0)}]")

    # Solve at arbitrary evaluation times (including unobserved)
    eval_times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    latent_states = model.latent_dynamics(z0, eval_times)

    print(f"\nPredictions at evaluation times:")
    for t, z in latent_states.items():
        pred = model.decode(z)
        true = np.sin(t)
        print(f"  t={t:.1f}: pred={pred[0]:.3f}, true={true:.3f}")
