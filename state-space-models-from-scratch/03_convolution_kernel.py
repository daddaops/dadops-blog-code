"""SSM dual computation: convolution kernel via FFT vs sequential recurrence.

Demonstrates that the convolution and recurrence produce identical outputs —
the mathematical identity that lets SSMs train in parallel (FFT) and
infer sequentially (recurrence).
"""
import numpy as np

# --- Reproduce continuous SSM and discretization from scripts 01-02 ---
N = 4
A = np.array([
    [-0.5,  1.0,  0.0,  0.0],
    [-1.0, -0.5,  0.0,  0.0],
    [ 0.0,  0.0, -0.1,  0.3],
    [ 0.0,  0.0, -0.3, -0.1]
])
B = np.array([[1.0], [0.0], [0.5], [0.0]])
C = np.array([[1.0, 0.0, 1.0, 0.0]])

def discretize_zoh(A, B, delta):
    N = A.shape[0]
    dA = delta * A
    A_bar = np.eye(N) + dA + dA @ dA / 2 + dA @ dA @ dA / 6
    B_bar = delta * B
    return A_bar, B_bar

delta = 0.1
A_bar, B_bar = discretize_zoh(A, B, delta)

L = 100
u_disc = np.zeros(L)
u_disc[10:20] = 1.0

# Sequential recurrence (already computed inline)
x_d = np.zeros((L, N))
y_d = np.zeros(L)
for k in range(1, L):
    x_d[k] = A_bar @ x_d[k-1] + B_bar.squeeze() * u_disc[k]
    y_d[k] = (C @ x_d[k]).item()

# --- Convolution kernel ---
def ssm_kernel(A_bar, B_bar, C, L):
    """Compute the SSM convolution kernel of length L.
    K[i] = C @ A_bar^i @ B_bar"""
    kernel = np.zeros(L)
    A_power = np.eye(A_bar.shape[0])  # A_bar^0 = I
    for i in range(L):
        kernel[i] = (C @ A_power @ B_bar).item()
        A_power = A_power @ A_bar
    return kernel

K = ssm_kernel(A_bar, B_bar, C, L)

# Method 1: Convolution via FFT (parallel — training mode)
# Pad to avoid circular convolution artifacts
pad_len = 2 * L
y_conv = np.real(np.fft.ifft(
    np.fft.fft(K, pad_len) * np.fft.fft(u_disc, pad_len)
))[:L]

# Method 2: Sequential recurrence (already computed as y_d)
max_diff = np.max(np.abs(y_conv - y_d))
print(f"Convolution vs recurrence max diff: {max_diff:.2e}")
print(f"Kernel K shape: {K.shape}")
print(f"K[0]={K[0]:.4f}, K[1]={K[1]:.4f}, K[2]={K[2]:.4f} — decaying impulse response")
print(f"Train: O(L log L) via FFT | Infer: O(1) per step via recurrence")
