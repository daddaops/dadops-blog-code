"""MAP vs MLE Estimation — polynomial fitting with and without a prior.

Demonstrates how MAP (L2 regularization = Gaussian prior) prevents
overfitting in high-degree polynomial regression.
"""
import numpy as np

# MAP vs MLE: Polynomial fitting with and without a prior
np.random.seed(42)

# Generate noisy data from a simple quadratic
x = np.linspace(-1, 1, 15)
y_true = 0.5 * x**2 - 0.3 * x + 0.1
y = y_true + np.random.normal(0, 0.15, len(x))

def fit_polynomial(x, y, degree, reg_lambda=0.0):
    """Fit polynomial with optional L2 regularization (= Gaussian prior)."""
    X = np.column_stack([x**d for d in range(degree + 1)])
    if reg_lambda > 0:
        # MAP with Gaussian prior: (X^T X + lambda I)^-1 X^T y
        w = np.linalg.solve(X.T @ X + reg_lambda * np.eye(degree + 1), X.T @ y)
    else:
        # MLE: (X^T X)^-1 X^T y
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

# Fit degree-10 polynomial — way too complex for 15 points
w_mle = fit_polynomial(x, y, degree=10, reg_lambda=0.0)
w_map = fit_polynomial(x, y, degree=10, reg_lambda=1.0)

# Evaluate on a fine grid
x_test = np.linspace(-1, 1, 200)
X_test = np.column_stack([x_test**d for d in range(11)])
y_truth = 0.5 * x_test**2 - 0.3 * x_test + 0.1

mle_mse = np.mean((X_test @ w_mle - y_truth)**2)
map_mse = np.mean((X_test @ w_map - y_truth)**2)

print(f"MLE test MSE: {mle_mse:.4f}")  # Overfits badly
print(f"MAP test MSE: {map_mse:.4f}")  # Smooth, close to truth

print(f"\nMLE max |weight|: {np.max(np.abs(w_mle)):.1f}")  # Huge weights
print(f"MAP max |weight|: {np.max(np.abs(w_map)):.1f}")   # Small weights
