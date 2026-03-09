"""Bayesian Linear Regression from Scratch — full posterior + predictive uncertainty.

Computes the closed-form posterior for linear regression weights and
demonstrates credible intervals that widen away from training data.
"""
import numpy as np

def bayesian_linear_regression(X, y, sigma_noise, sigma_prior):
    """Compute posterior mean and covariance for Bayesian linear regression."""
    n_features = X.shape[1]
    precision_noise = 1 / sigma_noise**2
    precision_prior = 1 / sigma_prior**2

    # Posterior covariance: (sigma_n^-2 * X^T X + sigma_w^-2 * I)^-1
    Sigma_post = np.linalg.inv(
        precision_noise * X.T @ X + precision_prior * np.eye(n_features)
    )
    # Posterior mean: sigma_n^-2 * Sigma_post @ X^T @ y
    mu_post = precision_noise * Sigma_post @ X.T @ y

    return mu_post, Sigma_post

def predict_with_uncertainty(X_new, mu_post, Sigma_post, sigma_noise):
    """Predict with credible intervals."""
    y_pred = X_new @ mu_post
    # Predictive variance = model uncertainty + noise
    pred_var = np.array([
        x @ Sigma_post @ x + sigma_noise**2 for x in X_new
    ])
    return y_pred, np.sqrt(pred_var)

# Generate data from y = 1.5x + 0.5 + noise
np.random.seed(42)
x_train = np.sort(np.random.uniform(-2, 2, 20))
y_train = 1.5 * x_train + 0.5 + np.random.normal(0, 0.5, 20)

# Design matrix with bias term: [1, x]
X_train = np.column_stack([np.ones_like(x_train), x_train])

mu_post, Sigma_post = bayesian_linear_regression(X_train, y_train, 0.5, 2.0)
print(f"Posterior weights: intercept={mu_post[0]:.2f}, slope={mu_post[1]:.2f}")
# Close to true values: intercept=0.5, slope=1.5

# Predict on a fine grid including extrapolation
x_test = np.linspace(-4, 4, 200)
X_test = np.column_stack([np.ones_like(x_test), x_test])
y_pred, y_std = predict_with_uncertainty(X_test, mu_post, Sigma_post, 0.5)

# Three things MLE can't give you:
# 1. Credible intervals: y_pred +/- 2*y_std covers 95% of predictions
# 2. Growing uncertainty: y_std is small near data, large far away
# 3. The posterior itself: Sigma_post tells you weight correlations
in_data = (x_test > -2) & (x_test < 2)
out_data = (x_test < -3) | (x_test > 3)
print(f"Avg uncertainty in data range:    {y_std[in_data].mean():.3f}")
print(f"Avg uncertainty in extrapolation: {y_std[out_data].mean():.3f}")
# Uncertainty is much larger when extrapolating — Bayesian knows what it doesn't know
