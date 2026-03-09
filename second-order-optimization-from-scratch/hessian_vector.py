"""Hessian-vector products, power iteration, and conjugate gradient."""
import numpy as np

def hessian_vector_product(grad_fn, theta, v, eps=1e-5):
    """Compute H @ v using central finite differences."""
    g_plus = grad_fn(theta + eps * v)
    g_minus = grad_fn(theta - eps * v)
    return (g_plus - g_minus) / (2 * eps)

def power_iteration(grad_fn, theta, num_iters=50):
    """Find the top eigenvalue of the Hessian via power iteration."""
    v = np.random.randn(len(theta))
    v = v / np.linalg.norm(v)
    for _ in range(num_iters):
        Hv = hessian_vector_product(grad_fn, theta, v)
        eigenvalue = v @ Hv
        v = Hv / np.linalg.norm(Hv)
    return eigenvalue, v

def conjugate_gradient(grad_fn, theta, grad, max_iters=20, tol=1e-8):
    """Solve H @ delta = -grad approximately using CG + Hv products."""
    delta = np.zeros_like(grad)
    r = -grad.copy()  # residual
    p = r.copy()       # search direction
    rs_old = r @ r

    for i in range(max_iters):
        Hp = hessian_vector_product(grad_fn, theta, p)
        alpha = rs_old / (p @ Hp + 1e-10)
        delta = delta + alpha * p
        r = r - alpha * Hp
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return delta

# Demo: 5D quadratic with known Hessian
np.random.seed(42)
eigvals = np.array([1.0, 5.0, 20.0, 50.0, 200.0])
Q, _ = np.linalg.qr(np.random.randn(5, 5))
A = Q @ np.diag(eigvals) @ Q.T

grad_fn = lambda x: A @ x
theta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Power iteration: find top eigenvalue
top_eigval, top_eigvec = power_iteration(grad_fn, theta)
print(f"Top eigenvalue (power iter): {top_eigval:.1f}  (true: 200.0)")

# CG: approximate Newton step
grad = grad_fn(theta)
cg_step = conjugate_gradient(grad_fn, theta, grad, max_iters=10)
newton_step = -np.linalg.solve(A, grad)  # exact Newton
print(f"CG step error vs exact Newton: {np.linalg.norm(cg_step - newton_step):.2e}")
# Output: CG matches exact Newton to high precision in ≤ n iterations
