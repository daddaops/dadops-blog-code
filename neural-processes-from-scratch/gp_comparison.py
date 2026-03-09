"""Compare GP (exact, O(n^3)) vs NP (approximate, O(n)) inference.

GP provides calibrated uncertainty but scales cubically.
NPs trade calibration for speed and amortization.
"""
import numpy as np
import time
from cnp import CNP, make_sine_task


def gp_predict(x_ctx, y_ctx, x_test, l=1.0, sigma_n=0.1):
    """Standard GP posterior with RBF kernel."""
    def rbf(x1, x2):
        return np.exp(-0.5 * ((x1 - x2.T) / l) ** 2)

    K = rbf(x_ctx, x_ctx) + sigma_n**2 * np.eye(len(x_ctx))
    K_s = rbf(x_ctx, x_test)
    K_ss = rbf(x_test, x_test)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_ctx))
    mu = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    sigma = np.sqrt(np.diag(K_ss - v.T @ v).clip(0))
    return mu.ravel(), sigma


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x_all, y_all = make_sine_task(rng)
    x_ctx, y_ctx = x_all[:5], y_all[:5]

    x_test = np.linspace(-4, 4, 200).reshape(-1, 1)

    cnp = CNP()

    # GP inference (exact, O(n^3))
    t0 = time.perf_counter()
    gp_mu, gp_sigma = gp_predict(x_ctx, y_ctx, x_test)
    gp_time = (time.perf_counter() - t0) * 1000

    # NP inference (approximate, O(n))
    t0 = time.perf_counter()
    np_mu, np_sigma = cnp.predict(x_ctx, y_ctx, x_test)
    np_time = (time.perf_counter() - t0) * 1000

    print(f"GP: {gp_time:.1f}ms | NP: {np_time:.1f}ms")
    print(f"GP uncertainty range: [{gp_sigma.min():.3f}, {gp_sigma.max():.3f}]")
    print(f"NP uncertainty range: [{np_sigma.min():.3f}, {np_sigma.max():.3f}]")
