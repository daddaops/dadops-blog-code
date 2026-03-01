"""
Sigmoid and tanh: the classic activation functions.

Computes both functions and their gradients, showing how
sigmoid's max gradient is only 0.25 (leading to vanishing gradients)
while tanh reaches 1.0 at the center.

Requires: numpy

From: https://dadops.dev/blog/activation-functions-from-scratch/
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(x):
    return 1 - np.tanh(x) ** 2

# Compare at a few points
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

print("x:             ", x)
print("sigmoid(x):    ", np.round(sigmoid(x), 4))
print("sigmoid'(x):   ", np.round(sigmoid_grad(x), 4))
print("tanh(x):       ", np.round(tanh(x), 4))
print("tanh'(x):      ", np.round(tanh_grad(x), 4))

# sigmoid'(x):  [0.1050, 0.1966, 0.2500, 0.1966, 0.1050]
# tanh'(x):     [0.0707, 0.4200, 1.0000, 0.4200, 0.0707]
