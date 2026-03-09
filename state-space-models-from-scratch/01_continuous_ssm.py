"""Continuous-time State Space Model simulation using Euler's method.

Simulates a 4D SSM with oscillatory dynamics (two damped rotation blocks)
driven by a pulse input.
"""
import numpy as np

# A 4-dimensional continuous-time SSM with oscillatory dynamics
N = 4  # state dimension

# State matrix: two 2x2 blocks, each a damped rotation
A = np.array([
    [-0.5,  1.0,  0.0,  0.0],   # block 1: fast oscillation
    [-1.0, -0.5,  0.0,  0.0],
    [ 0.0,  0.0, -0.1,  0.3],   # block 2: slow oscillation
    [ 0.0,  0.0, -0.3, -0.1]
])
B = np.array([[1.0], [0.0], [0.5], [0.0]])   # input coupling
C = np.array([[1.0, 0.0, 1.0, 0.0]])          # readout

# Simulate with Euler's method: x(t+dt) ≈ x(t) + dt * x'(t)
dt, T = 0.01, 10.0
steps = int(T / dt)
u = np.zeros(steps)
u[100:200] = 1.0   # pulse input from t=1.0 to t=2.0

x = np.zeros((steps, N))
y = np.zeros(steps)

for k in range(1, steps):
    dx = A @ x[k-1] + B.squeeze() * u[k-1]
    x[k] = x[k-1] + dt * dx
    y[k] = (C @ x[k]).item()

print(f"State at t=5.0: [{', '.join(f'{v:.3f}' for v in x[500])}]")
print(f"Output range:   [{y.min():.3f}, {y.max():.3f}]")
print(f"State decays after pulse ends — A's negative eigenvalues ensure stability")
