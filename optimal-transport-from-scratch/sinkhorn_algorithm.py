"""Sinkhorn's algorithm for entropy-regularized optimal transport.

Compares sharp (small epsilon) vs diffuse (large epsilon) transport plans.
"""
import numpy as np

def sinkhorn(cost_matrix, a, b, epsilon=0.1, max_iter=100, tol=1e-8):
    """Sinkhorn's algorithm for entropy-regularized optimal transport.
    cost_matrix: (n, m) pairwise costs
    a: (n,) source weights   b: (m,) target weights
    epsilon: regularization strength (smaller = closer to exact OT)
    Returns: transport plan gamma, list of iteration costs."""
    n, m = cost_matrix.shape
    K = np.exp(-cost_matrix / epsilon)   # Gibbs kernel

    u = np.ones(n)           # row scaling factors
    v = np.ones(m)           # column scaling factors
    costs = []

    for iteration in range(max_iter):
        u_prev = u.copy()
        u = a / (K @ v)          # row normalization
        v = b / (K.T @ u)        # column normalization

        # Transport plan and cost for monitoring
        gamma = np.diag(u) @ K @ np.diag(v)
        transport_cost = np.sum(gamma * cost_matrix)
        costs.append(transport_cost)

        # Check convergence
        if np.max(np.abs(u - u_prev)) < tol:
            break

    return gamma, costs

# Example: 5 source points, 5 target points
np.random.seed(42)
source = np.random.rand(5, 2)
target = np.random.rand(5, 2) + np.array([0.5, 0])
C = np.array([[np.sum((s - t) ** 2) for t in target] for s in source])
a = np.ones(5) / 5
b = np.ones(5) / 5

# Compare sharp vs diffuse plans
for eps in [0.01, 0.1, 1.0]:
    gamma, costs = sinkhorn(C, a, b, epsilon=eps)
    max_g = gamma.max()
    print(f"epsilon={eps:.2f}: cost={costs[-1]:.3f}, "
          f"converged in {len(costs)} iters")
    print(f"  Plan sparsity: {np.sum(gamma < max_g * 0.01)}/{gamma.size} near-zero entries")
