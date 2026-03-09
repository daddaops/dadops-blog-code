"""Gradient descent on an ill-conditioned quadratic."""
import numpy as np

# Ill-conditioned 2D quadratic: f(x) = 0.5 * x^T A x
# Eigenvalues 1 and 100 give condition number kappa = 100
A = np.array([[100.0, 0.0],
              [0.0,   1.0]])

def f(x):
    return 0.5 * x @ A @ x

def grad_f(x):
    return A @ x

# Gradient descent with fixed step size
x = np.array([1.0, 1.0])
lr = 0.019  # near-optimal for this problem: 2/(lambda_max + lambda_min)
trajectory = [x.copy()]

for i in range(200):
    x = x - lr * grad_f(x)
    trajectory.append(x.copy())
    if np.linalg.norm(grad_f(x)) < 1e-6:
        break

print(f"GD converged in {len(trajectory)-1} iterations")
print(f"Final point: ({x[0]:.6f}, {x[1]:.6f})")
# Output: GD converged in 200 iterations (didn't converge!)
# The zigzag pattern in x[0] shows the problem
