import numpy as np
np.random.seed(42)

# Generate a simple linear regression task
D = 3              # input dimension
C = 20             # number of in-context examples
w_true = np.array([2.0, -1.0, 0.5])  # true weights

X = np.random.randn(C, D)
y = X @ w_true + 0.1 * np.random.randn(C)  # y = Xw + noise

# --- NumPy closed-form least squares ---
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"Least squares:  w = {w_lstsq.round(3)}")

# --- Attention-as-gradient-descent ---
# Initialize implicit weights to zero
w = np.zeros(D)
eta = 0.1  # learning rate

# Each "layer" = one GD step via attention
for layer in range(200):
    predictions = X @ w                    # w^T x_i for all i
    errors = predictions - y               # prediction errors
    grad = (X.T @ errors) / C             # mean gradient
    w = w - eta * grad                     # gradient descent step

print(f"Attention (GD):  w = {w.round(3)}")
print(f"True weights:    w = {w_true}")
# Least squares:  w = [2.004, -0.999, 0.489]
# Attention (GD):  w = [2.004, -0.999, 0.489]
# True weights:    w = [2.0, -1.0, 0.5]
