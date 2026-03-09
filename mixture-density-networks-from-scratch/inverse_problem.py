"""The inverse problem: MSE regression fails on multimodal data.

Demonstrates that a standard neural network trained with MSE loss
predicts the conditional mean, which falls between modes for
multimodal inverse mappings.
"""
import numpy as np

# Forward function: y = x + 0.3*sin(2*pi*x) — single-valued
np.random.seed(42)
x = np.random.uniform(0, 1, 500)
y = x + 0.3 * np.sin(2 * np.pi * x) + np.random.normal(0, 0.03, 500)

# Invert: now predict x from y (one y can map to multiple x values)
X_inv, Y_inv = y.reshape(-1, 1), x  # input=y, target=x

# MSE regression: learns the conditional mean
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42)
mlp.fit(X_inv, Y_inv)

# For y=0.5, there are ~3 valid x values, but MSE predicts their average
y_query = np.array([[0.5]])
print(f"MSE prediction for y=0.5: x = {mlp.predict(y_query)[0]:.3f}")
print("True solutions: x ≈ 0.20, 0.50, 0.80 — the mean misses all three")
