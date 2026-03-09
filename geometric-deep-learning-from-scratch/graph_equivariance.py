import numpy as np

# 5-node path graph: 0-1-2-3-4
A = np.array([[0,1,0,0,0],
              [1,0,1,0,0],
              [0,1,0,1,0],
              [0,0,1,0,1],
              [0,0,0,1,0]], dtype=float)

# Node features: one-hot identity
X = np.eye(5)

# Random unconstrained weight matrix
W_random = np.random.RandomState(42).randn(5, 5)

# Permutation: reverse node ordering (0<->4, 1<->3, 2 stays)
P = np.eye(5)[::-1]

# Test equivariance: does W @ (P @ X) == P @ (W @ X)?
lhs = W_random @ (P @ X)
rhs = P @ (W_random @ X)
print(f"Random W equivariant? {np.allclose(lhs, rhs)}")  # False!

# Equivariant alternative: polynomial in A
c0, c1, c2 = 0.5, 0.3, 0.1
W_equiv = c0 * np.eye(5) + c1 * A + c2 * (A @ A)

# This graph is symmetric under reversal, so P @ A @ P.T == A
lhs = W_equiv @ (P @ X)
rhs = P @ (W_equiv @ X)
print(f"Poly(A) equivariant? {np.allclose(lhs, rhs)}")   # True!

# W_equiv IS message passing: node 2 aggregates from 0,1,2,3,4 hops
print(f"Node 2 receives from: {np.round(W_equiv[2], 2)}")
# [0.1, 0.3, 0.7, 0.3, 0.1] -- weighted by hop distance!
