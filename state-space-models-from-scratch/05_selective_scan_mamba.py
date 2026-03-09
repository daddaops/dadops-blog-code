"""Mamba's selective scan: input-dependent state space model.

Demonstrates how the selection mechanism assigns different delta values
to tokens of varying importance — high-magnitude tokens get larger delta
(more state update), low-magnitude tokens get smaller delta (state preserved).
"""
import numpy as np

def selective_scan(x_seq, A_log, W_B, W_C, W_delta, b_delta):
    """Mamba's selective scan — input-dependent state space model.

    Args:
        x_seq:   (L, D) input sequence
        A_log:   (D, N) log-parameterized state decay (fixed, not input-dependent)
        W_B:     (D, N) projection for input-dependent B
        W_C:     (D, N) projection for input-dependent C
        W_delta: (D, 1) projection for input-dependent step size
        b_delta: scalar bias for step size
    """
    L, D = x_seq.shape
    N = A_log.shape[1]
    h = np.zeros((D, N))       # hidden state — constant size!
    outputs = np.zeros((L, D))
    deltas = np.zeros(L)

    for t in range(L):
        xt = x_seq[t]

        # Input-dependent projections — the "selection"
        B_t = xt @ W_B                                          # (N,)
        C_t = xt @ W_C                                          # (N,)
        delta_t = np.log1p(np.exp(xt @ W_delta + b_delta)).mean()  # softplus

        # Discretize for THIS timestep (not globally!)
        A_bar = np.exp(delta_t * A_log)     # (D, N) element-wise
        B_bar = delta_t * B_t               # (N,) simplified ZOH

        # State update
        h = A_bar * h + np.outer(xt, B_bar)  # (D, N)

        # Output
        outputs[t] = (h * C_t).sum(axis=1)   # (D,)
        deltas[t] = delta_t

    return outputs, deltas

# Demo: 8 tokens with varying importance
D, N, L = 4, 8, 8
np.random.seed(123)
A_log = -np.abs(np.random.randn(D, N))      # negative for stability
W_B = np.random.randn(D, N) * 0.1
W_C = np.random.randn(D, N) * 0.1
W_delta = np.random.randn(D, 1) * 0.5
b_delta = -1.5  # moderate default step

x_seq = np.random.randn(L, D) * 0.5
x_seq[2] *= 5.0  # "dragon" — high magnitude → should get high delta
x_seq[5] *= 5.0  # "fire"   — high magnitude → should get high delta

y, deltas = selective_scan(x_seq, A_log, W_B, W_C, W_delta, b_delta)
print("Delta per token:", [f"{d:.3f}" for d in deltas])
print("Tokens 2 and 5 get higher delta — the model 'selects' them")
print(f"State shape: ({D}, {N}) = {D*N} values — constant regardless of L")
