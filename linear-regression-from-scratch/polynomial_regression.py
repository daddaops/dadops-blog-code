import numpy as np

# Nonlinear ground truth: y = sin(x) + noise
np.random.seed(42)
X_train = np.sort(np.random.uniform(0, 2 * np.pi, 20))
y_train = np.sin(X_train) + np.random.randn(20) * 0.3
X_test = np.linspace(0, 2 * np.pi, 200)
y_test_true = np.sin(X_test)

def poly_features(x, degree):
    return np.column_stack([x ** i for i in range(degree + 1)])

print(f"{'Degree':>6} {'Train MSE':>12} {'Test MSE':>12} {'Max |coeff|':>14}")
print("-" * 48)

for degree in [1, 3, 5, 10, 15]:
    X_poly = poly_features(X_train, degree)
    w = np.linalg.pinv(X_poly) @ y_train

    train_pred = X_poly @ w
    test_pred = poly_features(X_test, degree) @ w

    train_mse = np.mean((y_train - train_pred) ** 2)
    test_mse = np.mean((y_test_true - test_pred) ** 2)

    print(f"{degree:>6} {train_mse:>12.6f} {test_mse:>12.6f} {np.max(np.abs(w)):>14.1f}")

# Degree 15: enormous coefficients = overfitting fingerprint
