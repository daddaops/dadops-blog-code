"""Kantorovich optimal transport via linear programming.

Allows mass splitting — source mass can be distributed
across multiple targets (unlike Monge's 1-to-1 assignment).
"""
import numpy as np
from scipy.optimize import linprog

def kantorovich_transport(source_pts, target_pts, a, b):
    """Solve the Kantorovich optimal transport problem via LP.
    a = source weights (sum to 1), b = target weights (sum to 1)."""
    n, m = len(source_pts), len(target_pts)

    # Cost matrix (squared Euclidean distances)
    C = np.array([[np.sum((s - t) ** 2) for t in target_pts]
                   for s in source_pts])

    # Flatten cost for LP: min c^T x where x = gamma.ravel()
    c_vec = C.ravel()

    # Equality constraints: row sums = a, col sums = b
    # Row sums: for each source i, sum_j gamma_ij = a_i
    A_row = np.zeros((n, n * m))
    for i in range(n):
        A_row[i, i * m:(i + 1) * m] = 1.0

    # Col sums: for each target j, sum_i gamma_ij = b_j
    A_col = np.zeros((m, n * m))
    for j in range(m):
        for i in range(n):
            A_col[j, i * m + j] = 1.0

    A_eq = np.vstack([A_row, A_col])
    b_eq = np.concatenate([a, b])

    result = linprog(c_vec, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, None)] * (n * m), method='highs')

    gamma = result.x.reshape(n, m)
    return gamma, result.fun

# Unequal masses: 3 sources, 4 targets (mass must split!)
source = np.array([[0, 0], [1, 0], [0.5, 1]])
target = np.array([[2, 0], [3, 0], [2, 1], [3, 1]])
a = np.array([0.5, 0.3, 0.2])        # source weights
b = np.array([0.2, 0.3, 0.2, 0.3])   # target weights

gamma, cost = kantorovich_transport(source, target, a, b)
print(f"Optimal transport cost: {cost:.4f}")
print(f"\nCoupling matrix gamma (rows=source, cols=target):")
print(np.round(gamma, 3))
# Non-zero off-diagonal entries show mass splitting
