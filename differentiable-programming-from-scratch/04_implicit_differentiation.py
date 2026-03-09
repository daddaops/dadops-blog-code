import numpy as np

def solve_cube_root(p, tol=1e-12):
    """Find x such that x^3 - p = 0, using Newton's method."""
    x = float(p) ** (1/3) if p > 0 else -(abs(float(p)) ** (1/3))
    for _ in range(100):
        g = x ** 3 - float(p)
        dg_dx = 3 * x ** 2
        if abs(dg_dx) < 1e-15:
            break
        x = x - g / dg_dx
        if abs(g) < tol:
            break
    return x

def implicit_grad(p):
    """Compute dx*/dp via the Implicit Function Theorem.
    g(p, x) = x^3 - p = 0
    dg/dx = 3x^2,  dg/dp = -1
    dx*/dp = -(dg/dx)^{-1} * (dg/dp) = -1/(3x^2) * (-1) = 1/(3x^2)
    """
    x_star = solve_cube_root(p)
    dg_dx = 3 * x_star ** 2    # partial of g w.r.t. x at solution
    dg_dp = -1.0                # partial of g w.r.t. p at solution
    return x_star, -dg_dp / dg_dx

# Test: d(p^{1/3})/dp = (1/3) * p^{-2/3}
for p in [1.0, 8.0, 27.0]:
    x_star, grad = implicit_grad(p)
    analytic = (1/3) * p ** (-2/3)
    print(f"p={p:5.1f}  x*={x_star:.6f}  "
          f"IFT grad={grad:.6f}  analytic={analytic:.6f}")
# p=  1.0  x*=1.000000  IFT grad=0.333333  analytic=0.333333
# p=  8.0  x*=2.000000  IFT grad=0.083333  analytic=0.083333
# p= 27.0  x*=3.000000  IFT grad=0.037037  analytic=0.037037
