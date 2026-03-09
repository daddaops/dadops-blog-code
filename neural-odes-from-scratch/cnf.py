"""Continuous Normalizing Flows with Hutchinson trace estimator.

Solves the CNF: dz/dt = f(z,t), d log p/dt = -tr(df/dz).
No invertibility constraint, no determinant computation.
"""
import numpy as np
from neural_ode import NeuralODE


def hutchinson_trace(f, h, t, dim, n_probes=5):
    """Estimate tr(df/dh) via Hutchinson's trick: E[v^T J v]."""
    eps = 1e-5
    trace_est = 0.0
    for _ in range(n_probes):
        v = np.random.randn(dim)
        jvp = (f(h + eps * v, t) - f(h - eps * v, t)) / (2 * eps)
        trace_est += np.dot(v, jvp)
    return trace_est / n_probes


def cnf_forward(node, z0, t0=0.0, t1=1.0, steps=50):
    """Solve the CNF: dz/dt = f(z,t), d log p/dt = -tr(df/dz)."""
    dt = (t1 - t0) / steps
    z = z0.copy()
    log_p_change = 0.0

    for i in range(steps):
        t = t0 + i * dt
        dz = node.f(z, t)
        trace = hutchinson_trace(node.f, z, t, len(z))
        log_p_change -= trace * dt  # Liouville's equation
        z = z + dt * dz

    return z, log_p_change


if __name__ == "__main__":
    np.random.seed(42)
    node = NeuralODE(dim=2, hidden=8)

    z0 = np.array([0.5, -0.3])
    z_final, log_p = cnf_forward(node, z0)

    print(f"Input:  [{z0[0]:.3f}, {z0[1]:.3f}]")
    print(f"Output: [{z_final[0]:.3f}, {z_final[1]:.3f}]")
    print(f"Log density change: {log_p:.4f}")

    # Compare Hutchinson trace estimate to exact trace
    h = z0.copy()
    exact_J = np.zeros((2, 2))
    eps = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        exact_J[:, j] = (node.f(h + e, 0) - node.f(h - e, 0)) / (2 * eps)
    exact_trace = np.trace(exact_J)
    hutch_trace = hutchinson_trace(node.f, h, 0, 2, n_probes=1000)
    print(f"\nExact trace:      {exact_trace:.4f}")
    print(f"Hutchinson (1000): {hutch_trace:.4f}")
