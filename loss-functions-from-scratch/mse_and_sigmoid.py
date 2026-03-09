import numpy as np

def mse_loss(predictions, targets):
    """Mean squared error: average of squared residuals."""
    return np.mean((predictions - targets) ** 2)

# A model predicting house prices (in $100k)
predictions = np.array([2.5, 0.0, 2.1, 7.8])
targets     = np.array([3.0, -0.5, 2.0, 7.5])

loss = mse_loss(predictions, targets)
print(f"MSE: {loss:.4f}")
# MSE: 0.1500

# Gradient of MSE w.r.t. a single prediction
# dMSE/dy_hat = (2/n)(y_hat - y)

def mse_gradient(prediction, target, n):
    """Gradient of MSE with respect to prediction."""
    return (2 / n) * (prediction - target)

# If we predicted 2.5 but the target was 3.0:
grad = mse_gradient(2.5, 3.0, n=4)
print(f"Gradient: {grad:.4f}")
# Gradient: -0.2500
# Negative -> push the prediction UP toward 3.0. Makes sense!

def sigmoid(z):
    """Squash any real number into (0, 1)."""
    return 1 / (1 + np.exp(-z))

# Model's raw output (logit) and true label
z = -4.5     # model's raw score
y = 1        # true label: this IS spam

p = sigmoid(z)  # predicted probability
print(f"Prediction: {p:.6f}")
# Prediction: 0.010987
# Model says 1.1% chance of spam. True label is 1. It's VERY wrong.

# MSE on the probability
mse = (p - y) ** 2
print(f"MSE loss: {mse:.6f}")
# MSE loss: 0.978147
# Loss is high (close to 1). Good — the model is very wrong.

# To update the model's weights, we need dLoss/dz (gradient w.r.t. the logit)
# Chain rule: dMSE/dz = dMSE/dp * dp/dz
#   dMSE/dp = 2(p - y)
#   dp/dz = sigmoid'(z) = p(1 - p)
# So: dMSE/dz = 2(p - y) * p(1 - p)

def sigmoid_derivative(p):
    return p * (1 - p)

def mse_grad_wrt_logit(p, y):
    """Gradient of MSE w.r.t. the pre-sigmoid logit z."""
    return 2 * (p - y) * sigmoid_derivative(p)

grad = mse_grad_wrt_logit(p, y)
print(f"Prediction: {p:.4f} | Target: {y} | MSE gradient: {grad:.6f}")
# Prediction: 0.0110 | Target: 1 | MSE gradient: -0.021494
