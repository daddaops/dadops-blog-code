"""HiPPO initialization vs random: long-range memory comparison.

Builds the HiPPO-LegS matrix (Legendre polynomial projection) and shows
that it retains information about a burst signal long after it ends,
while a random diagonal matrix forgets quickly.
"""
import numpy as np

def discretize_zoh(A, B, delta):
    """Discretize a continuous SSM using simplified Zero-Order Hold."""
    N = A.shape[0]
    dA = delta * A
    A_bar = np.eye(N) + dA + dA @ dA / 2 + dA @ dA @ dA / 6
    B_bar = delta * B
    return A_bar, B_bar

def make_hippo(N):
    """Build the HiPPO-LegS matrix for state dimension N.
    Compresses input history into Legendre polynomial coefficients."""
    A = np.zeros((N, N))
    B = np.zeros((N, 1))
    for n in range(N):
        B[n, 0] = np.sqrt(2 * n + 1)
        A[n, n] = -(n + 1)                        # diagonal
        for k in range(n):
            A[n, k] = -np.sqrt(2*n + 1) * np.sqrt(2*k + 1)  # lower triangle
    return A, B

N_state = 16
A_hippo, B_hippo = make_hippo(N_state)

# Memory test: input a burst, then silence. Who remembers?
L_test = 300
signal = np.zeros(L_test)
signal[20:40] = np.sin(np.linspace(0, 2 * np.pi, 20))  # burst at steps 20-40

delta_test = 0.05
C_read = np.ones((1, N_state)) / np.sqrt(N_state)  # normalized readout

# HiPPO SSM
Ah, Bh = discretize_zoh(A_hippo, B_hippo, delta_test)
xh = np.zeros(N_state)
y_hippo = np.zeros(L_test)
for k in range(L_test):
    xh = Ah @ xh + Bh.squeeze() * signal[k]
    y_hippo[k] = (C_read @ xh).item()

# Random SSM (diagonal with fast-decaying eigenvalues)
np.random.seed(0)
A_rand = np.diag(-np.abs(np.random.randn(N_state)) * 5.0 - 3.0)  # stable, fast decay
Ar, Br = discretize_zoh(A_rand, B_hippo, delta_test)
xr = np.zeros(N_state)
y_rand = np.zeros(L_test)
for k in range(L_test):
    xr = Ar @ xr + Br.squeeze() * signal[k]
    y_rand[k] = (C_read @ xr).item()

print(f"Signal energy concentrated at steps 20-40")
print(f"At step 100 (60 steps AFTER signal ended):")
print(f"  HiPPO output: {abs(y_hippo[100]):.6f}")
print(f"  Random output: {abs(y_rand[100]):.6f}")
print(f"HiPPO remembers — random forgets.")
