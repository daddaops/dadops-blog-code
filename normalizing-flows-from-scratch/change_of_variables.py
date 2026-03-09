"""Change of variables in 1D.

Demonstrates exact density computation using the change of variables
formula: p(x) = p_base(z) / |dx/dz|.
"""
import numpy as np


def change_of_variables_1d():
    """Demonstrate exact density computation via change of variables in 1D."""
    # Invertible transformation: x = f(z) = z + 0.4*sin(2z)
    # f'(z) = 1 + 0.8*cos(2z) >= 0.2 > 0 for all z -- guaranteed invertible
    def f(z):
        return z + 0.4 * np.sin(2 * z)

    def f_inv_newton(x, n_iter=20):
        """Invert f using Newton's method: find z such that f(z) = x."""
        z = x.copy()
        for _ in range(n_iter):
            z = z - (f(z) - x) / (1 + 0.8 * np.cos(2 * z))
        return z

    def df_dz(z):
        """Derivative of f: df/dz = 1 + 0.8*cos(2z)."""
        return 1 + 0.8 * np.cos(2 * z)

    # Exact density at a grid of points using change of variables
    x_grid = np.linspace(-4, 4, 200)
    z_grid = f_inv_newton(x_grid)
    base_density = np.exp(-0.5 * z_grid**2) / np.sqrt(2 * np.pi)
    # p(x) = p_base(z) * |dz/dx| = p_base(z) / |dx/dz|
    exact_density = base_density / np.abs(df_dz(z_grid))

    # Evaluate at specific points
    for test_x in [0.0, 1.5]:
        test_z = f_inv_newton(np.array([test_x]))[0]
        base_d = np.exp(-0.5 * test_z**2) / np.sqrt(2 * np.pi)
        exact_d = base_d / np.abs(df_dz(test_z))
        print(f"At x={test_x}: z={test_z:.3f}, exact p(x)={exact_d:.4f}")


if __name__ == "__main__":
    change_of_variables_1d()
