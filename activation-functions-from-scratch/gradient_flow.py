"""
Gradient flow simulation through deep layers.

Demonstrates how gradients vanish exponentially with sigmoid,
partially vanish with tanh, and flow perfectly with ReLU.

Requires: numpy

From: https://dadops.dev/blog/activation-functions-from-scratch/
"""

import numpy as np

def simulate_gradient_flow(activation_grad_fn, n_layers=10, input_val=0.5):
    """Simulate backward gradient flow through n layers."""
    gradient = 1.0  # start with gradient = 1 at the output
    layer_gradients = []

    for i in range(n_layers):
        local_grad = activation_grad_fn(input_val)
        gradient *= local_grad
        layer_gradients.append(gradient)

    return layer_gradients

# Sigmoid: gradient vanishes exponentially
sig_grads = simulate_gradient_flow(
    lambda x: 0.25,  # sigmoid's max gradient
    n_layers=10
)

# Tanh: better, but still vanishes
tanh_grads = simulate_gradient_flow(
    lambda x: 0.42,  # tanh's typical gradient near center
    n_layers=10
)

# ReLU: constant gradient (preview of the next section!)
relu_grads = simulate_gradient_flow(
    lambda x: 1.0,   # ReLU gradient for positive inputs
    n_layers=10
)

print("Layer:    ", [f"L{i+1:2d}" for i in range(10)])
print(f"Sigmoid:  {['%.2e' % g for g in sig_grads]}")
print(f"Tanh:     {['%.2e' % g for g in tanh_grads]}")
print(f"ReLU:     {['%.2e' % g for g in relu_grads]}")

# Sigmoid L10: 9.54e-07  (vanished)
# Tanh    L10: 1.73e-04  (nearly vanished)
# ReLU    L10: 1.00e+00  (perfect)
