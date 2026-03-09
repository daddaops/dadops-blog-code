"""Episodic meta-learning training loop for CNP.

Each episode: sample a random function, split into context/target,
compute NLL loss. The model learns to generalize across function families.
"""
import numpy as np
from cnp import CNP


def train_cnp(cnp, n_episodes=2000, lr=1e-3):
    """Episodic meta-learning training loop."""
    rng = np.random.default_rng(42)
    for ep in range(n_episodes):
        # Step 1: Sample a random function (sine with random params)
        amp = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 2.0)
        f = lambda x, a=amp, p=phase, fr=freq: a * np.sin(fr * x + p)

        # Step 2: Sample points from this function
        x_all = rng.uniform(-4, 4, size=(20, 1))
        y_all = f(x_all) + rng.normal(0, 0.05, size=x_all.shape)

        # Step 3: Random context/target split
        n_ctx = rng.integers(3, 15)
        idx = rng.permutation(20)
        x_ctx, y_ctx = x_all[idx[:n_ctx]], y_all[idx[:n_ctx]]
        x_tgt, y_tgt = x_all[idx[n_ctx:]], y_all[idx[n_ctx:]]

        # Step 4: Forward pass + NLL loss
        mu, sigma = cnp.predict(x_ctx, y_ctx, x_tgt)
        nll = ((y_tgt.ravel() - mu)**2 / (2 * sigma**2)
               + np.log(sigma)).mean()

        # Step 5: Update (in practice, use autograd + Adam)
        # Here we just track the loss for demonstration
        if ep % 500 == 0:
            print(f"Episode {ep:4d}  NLL: {nll:.3f}")


if __name__ == "__main__":
    cnp = CNP()
    train_cnp(cnp)
