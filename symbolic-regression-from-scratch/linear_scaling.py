"""Shared linear scaling fitness function."""
import numpy as np


def linear_scale_fitness(tree, x_data, y_data, alpha=0.001):
    """Evaluate fitness with linear scaling: y_hat = a * f(x) + b."""
    try:
        f = tree.evaluate(x_data)
        if np.any(np.isnan(f)) or np.any(np.isinf(f)):
            return 1e12
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
