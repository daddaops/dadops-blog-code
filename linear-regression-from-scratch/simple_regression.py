import numpy as np

# Generate noisy linear data: y = 3x + 7 + noise
np.random.seed(42)
X = 2 * np.random.rand(100)
y = 3 * X + 7 + np.random.randn(100) * 0.8

# Closed-form solution — no libraries needed
x_mean, y_mean = X.mean(), y.mean()
w = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
b = y_mean - w * x_mean

print(f"Slope:     {w:.4f}  (true: 3.0)")
print(f"Intercept: {b:.4f}  (true: 7.0)")

# R-squared: proportion of variance explained
y_pred = w * X + b
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R²:        {r_squared:.4f}")

# Sanity check against sklearn
from sklearn.linear_model import LinearRegression
sk_model = LinearRegression().fit(X.reshape(-1, 1), y)
print(f"\nsklearn slope: {sk_model.coef_[0]:.4f}, intercept: {sk_model.intercept_:.4f}")
print("Match: ✓" if abs(w - sk_model.coef_[0]) < 1e-10 else "Mismatch!")
