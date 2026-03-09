"""Shared data generation for regularization scripts."""
import numpy as np

np.random.seed(42)

# Generate 50 noisy sine points (train) and 200 clean points (test)
X_train = np.random.uniform(-3, 3, (50, 1))
y_train = np.sin(X_train) + np.random.randn(50, 1) * 0.3
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
y_test = np.sin(X_test)
