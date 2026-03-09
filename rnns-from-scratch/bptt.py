"""Backpropagation Through Time for a vanilla RNN."""
import numpy as np
from vanilla_rnn import RNNCell, VanillaRNN


def bptt(cell, xs, hiddens, d_outputs, W_out):
    """Backpropagation Through Time for a vanilla RNN.
    Returns gradients for all parameters.
    """
    T = len(xs)
    hidden_dim = cell.hidden_dim
    input_dim = xs[0].shape[0]

    # Initialize gradients
    dW_xh = np.zeros_like(cell.W_xh)
    dW_hh = np.zeros_like(cell.W_hh)
    db_h = np.zeros_like(cell.b_h)
    dW_out = np.zeros((hidden_dim, d_outputs[0].shape[0]))
    db_out = np.zeros(d_outputs[0].shape[0])

    # Gradient flowing back from future timesteps
    dh_next = np.zeros(hidden_dim)

    for t in reversed(range(T)):
        # Gradient from output at this timestep
        dy = d_outputs[t]                           # (output_dim,)
        dW_out += hiddens[t][:, None] @ dy[None, :]
        db_out += dy

        # Gradient into hidden state: from output + from future
        dh = dy @ W_out.T + dh_next                 # (hidden_dim,)

        # Backprop through tanh: d/dz = (1 - tanh^2(z)) * dh
        dtanh = (1 - hiddens[t] ** 2) * dh          # (hidden_dim,)

        # Gradients for weights
        dW_xh += xs[t][:, None] @ dtanh[None, :]
        h_prev = hiddens[t-1] if t > 0 else np.zeros(hidden_dim)
        dW_hh += h_prev[:, None] @ dtanh[None, :]
        db_h += dtanh

        # Pass gradient to previous timestep
        dh_next = dtanh @ cell.W_hh.T

    return dW_xh, dW_hh, db_h, dW_out, db_out


# Smoke test: run forward then backward
np.random.seed(42)
rnn = VanillaRNN(input_dim=3, hidden_dim=8, output_dim=4)
xs = [np.random.randn(3) for _ in range(5)]
outputs, hiddens = rnn.forward(xs)

# Fake output gradients (as if from a loss function)
d_outputs = [np.random.randn(4) for _ in range(5)]
grads = bptt(rnn.cell, xs, hiddens, d_outputs, rnn.W_out)
print(f"dW_xh shape: {grads[0].shape}")
print(f"dW_hh shape: {grads[1].shape}")
print(f"db_h shape:  {grads[2].shape}")
print(f"dW_out shape: {grads[3].shape}")
print(f"db_out shape: {grads[4].shape}")
print("BPTT completed successfully")
