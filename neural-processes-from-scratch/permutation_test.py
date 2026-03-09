"""Verify permutation invariance of the CNP encoder.

Mean pooling guarantees that context order doesn't matter.
"""
import numpy as np
from cnp import CNP, make_sine_task


def verify_permutation_invariance(cnp, x_ctx, y_ctx, x_target):
    """Show that shuffling context gives identical predictions."""
    mu1, sigma1 = cnp.predict(x_ctx, y_ctx, x_target)

    # Shuffle context points
    perm = np.random.permutation(len(x_ctx))
    x_shuf, y_shuf = x_ctx[perm], y_ctx[perm]
    mu2, sigma2 = cnp.predict(x_shuf, y_shuf, x_target)

    print(f"Original order  mu[:3]: {mu1[:3].round(4)}")
    print(f"Shuffled order  mu[:3]: {mu2[:3].round(4)}")
    print(f"Max difference: {np.max(np.abs(mu1 - mu2)):.2e}")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x_all, y_all = make_sine_task(rng)
    x_ctx, y_ctx = x_all[:5], y_all[:5]
    x_tgt = x_all[5:]

    cnp = CNP()
    verify_permutation_invariance(cnp, x_ctx, y_ctx, x_tgt)
