"""Numerical gradient checking — finite differences as ground truth.

Validates the chain rule by comparing hand-derived analytic gradients
against numerical finite differences for a composite function and a
neural network weight gradient.
"""
import numpy as np

def numerical_gradient(f, x, eps=1e-7):
    """Compute df/dx using central finite differences.
    This is the 'ground truth' that backprop must match."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Test 1: f(x) = sin(x^2 + 3x) at x=2
def f(x):
    return np.sin(x**2 + 3*x)

def f_grad_analytic(x):
    """Hand-derived: df/dx = cos(x^2 + 3x) * (2x + 3)"""
    return np.cos(x**2 + 3*x) * (2*x + 3)

x = 2.0
num_grad = numerical_gradient(f, x)
ana_grad = f_grad_analytic(x)
print(f"Numerical gradient:  {num_grad:.10f}")
print(f"Analytic gradient:   {ana_grad:.10f}")
print(f"Difference:          {abs(num_grad - ana_grad):.2e}")

# Test 2: Gradient of a neural network weight
# Simple: y = sigmoid(w*x + b), L = (y - target)^2
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x_val, target = 1.5, 1.0
w, b = 0.8, -0.2

def loss_fn(w_val):
    z = w_val * x_val + b
    y = sigmoid(z)
    return (y - target) ** 2

# Numerical gradient for w
num_dw = numerical_gradient(loss_fn, w)

# Analytic gradient via chain rule:
# dL/dw = dL/dy * dy/dz * dz/dw
z = w * x_val + b
y = sigmoid(z)
dL_dy = 2 * (y - target)           # loss gradient
dy_dz = y * (1 - y)                # sigmoid gradient
dz_dw = x_val                      # linear gradient
ana_dw = dL_dy * dy_dz * dz_dw

print(f"\nNeural net weight gradient:")
print(f"Numerical:  {num_dw:.10f}")
print(f"Analytic:   {ana_dw:.10f}")
print(f"Difference: {abs(num_dw - ana_dw):.2e}")

if __name__ == "__main__":
    pass  # Code runs at import time for simplicity
