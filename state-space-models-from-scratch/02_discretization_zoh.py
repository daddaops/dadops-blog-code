"""Zero-Order Hold discretization of a continuous SSM.

Converts continuous A,B matrices to discrete A_bar, B_bar using a Padé
approximation to the matrix exponential, then verifies the discrete
recurrence matches the continuous Euler simulation.
"""
import numpy as np

# --- Reproduce the continuous SSM from script 01 ---
N = 4
A = np.array([
    [-0.5,  1.0,  0.0,  0.0],
    [-1.0, -0.5,  0.0,  0.0],
    [ 0.0,  0.0, -0.1,  0.3],
    [ 0.0,  0.0, -0.3, -0.1]
])
B = np.array([[1.0], [0.0], [0.5], [0.0]])
C = np.array([[1.0, 0.0, 1.0, 0.0]])

dt_fine, T = 0.01, 10.0
steps = int(T / dt_fine)
u_fine = np.zeros(steps)
u_fine[100:200] = 1.0

x = np.zeros((steps, N))
y = np.zeros(steps)
for k in range(1, steps):
    dx = A @ x[k-1] + B.squeeze() * u_fine[k-1]
    x[k] = x[k-1] + dt_fine * dx
    y[k] = (C @ x[k]).item()

# --- Discretization ---
def discretize_zoh(A, B, delta):
    """Discretize a continuous SSM using simplified Zero-Order Hold.
    Returns discrete matrices A_bar, B_bar."""
    N = A.shape[0]
    # Exact: A_bar = expm(delta * A). We use a Padé approximation:
    # For small delta*A, expm ≈ I + dA + (dA)^2/2 + (dA)^3/6
    dA = delta * A
    A_bar = np.eye(N) + dA + dA @ dA / 2 + dA @ dA @ dA / 6
    B_bar = delta * B   # simplified ZOH (Mamba-style)
    return A_bar, B_bar

# Discretize with step size delta = 0.1 (10x coarser than Euler sim)
delta = 0.1
A_bar, B_bar = discretize_zoh(A, B, delta)

# Run discrete recurrence on the same pulse signal
L = 100  # 100 steps at delta=0.1 covers T=10.0
u_disc = np.zeros(L)
u_disc[10:20] = 1.0  # pulse from t=1.0 to t=2.0 (same as continuous)

x_d = np.zeros((L, N))
y_d = np.zeros(L)

for k in range(1, L):
    x_d[k] = A_bar @ x_d[k-1] + B_bar.squeeze() * u_disc[k]
    y_d[k] = (C @ x_d[k]).item()

# Compare with continuous simulation (sampled at same timesteps)
y_continuous_sampled = y[::10][:L]  # sample every 10th point
max_err = np.max(np.abs(y_d[:len(y_continuous_sampled)] - y_continuous_sampled))
print(f"Max error vs continuous: {max_err:.6f}")
print(f"Discrete state shape:   {x_d.shape} — just an RNN!")
print(f"Each step: multiply {N}x{N} matrix + add input. That's it.")
