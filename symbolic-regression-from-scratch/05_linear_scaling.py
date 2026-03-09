"""Linear scaling for fitness evaluation."""
import numpy as np
from node import Node
from tree_gen import init_population


def linear_scale_fitness(tree, x_data, y_data, alpha=0.001):
    """Evaluate fitness with linear scaling: y_hat = a * f(x) + b."""
    try:
        f = tree.evaluate(x_data)
        if np.any(np.isnan(f)) or np.any(np.isinf(f)):
            return 1e12
        # Closed-form least squares for a, b
        f_mean, y_mean = np.mean(f), np.mean(y_data)
        cov = np.mean((f - f_mean) * (y_data - y_mean))
        var_f = np.mean((f - f_mean) ** 2)
        if var_f < 1e-12:
            return 1e12
        a = cov / var_f
        b = y_mean - a * f_mean
        y_pred = a * f + b
        mse = np.mean((y_pred - y_data) ** 2)
    except Exception:
        return 1e12
    return mse + alpha * tree.size()


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    x = np.linspace(-3, 3, 50)
    y = 5.0 * x ** 2 + 3.0  # target: 5x^2 + 3

    # Build a tree that computes x^2 (without the scaling constants)
    tree = Node('*', [Node('x'), Node('x')])
    print("Expression:", tree)
    print("Target: 5*x^2 + 3")

    # Without linear scaling
    y_pred = tree.evaluate(x)
    mse_raw = np.mean((y_pred - y) ** 2)
    print(f"\nRaw MSE (no scaling): {mse_raw:.4f}")

    # With linear scaling
    fit = linear_scale_fitness(tree, x, y)
    print(f"Linear-scaled fitness: {fit:.6f}")

    # Show the learned a, b
    f = tree.evaluate(x)
    f_mean, y_mean = np.mean(f), np.mean(y)
    cov = np.mean((f - f_mean) * (y - y_mean))
    var_f = np.mean((f - f_mean) ** 2)
    a = cov / var_f
    b = y_mean - a * f_mean
    print(f"Learned scaling: a={a:.4f}, b={b:.4f}")
    print(f"(Expected: a=5.0, b=3.0)")
