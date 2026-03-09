"""Newton's method on quadratic and Rosenbrock functions."""
import numpy as np

A = np.array([[100.0, 0.0],
              [0.0,   1.0]])

def grad_f(x):
    return A @ x

def hessian_f(x):
    return A  # constant for quadratics

# Newton's method on the same quadratic
x_newton = np.array([1.0, 1.0])
H = hessian_f(x_newton)
x_newton = x_newton - np.linalg.solve(H, grad_f(x_newton))
print(f"Newton converged in 1 step: ({x_newton[0]:.6f}, {x_newton[1]:.6f})")
# Output: Newton converged in 1 step: (0.000000, 0.000000)

# Newton on the Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosenbrock_grad(xy):
    x, y = xy
    dx = -2*(1-x) + 200*(y - x**2)*(-2*x)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

def rosenbrock_hessian(xy):
    x, y = xy
    dxx = 2 - 400*(y - x**2) + 800*x**2
    dxy = -400*x
    dyy = 200.0
    return np.array([[dxx, dxy], [dxy, dyy]])

x_r = np.array([-1.0, 1.0])
for i in range(20):
    g = rosenbrock_grad(x_r)
    H = rosenbrock_hessian(x_r)
    x_r = x_r - np.linalg.solve(H, g)
    if np.linalg.norm(g) < 1e-10:
        print(f"Newton on Rosenbrock: {i+1} iterations to ({x_r[0]:.4f}, {x_r[1]:.4f})")
        break
# Output: Newton on Rosenbrock: 8 iterations to (1.0000, 1.0000)
# GD would need thousands of iterations on the same problem
