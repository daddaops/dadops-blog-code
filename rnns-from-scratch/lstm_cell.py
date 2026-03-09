"""LSTM cell with forget, input, candidate, and output gates."""
import numpy as np
from helpers import sigmoid


class LSTMCell:
    """A single LSTM cell with forget, input, and output gates."""
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = np.sqrt(2.0 / (concat_dim + hidden_dim))

        # Four gate weight matrices (concatenated input [h, x])
        self.W_f = np.random.randn(concat_dim, hidden_dim) * scale  # forget
        self.W_i = np.random.randn(concat_dim, hidden_dim) * scale  # input
        self.W_c = np.random.randn(concat_dim, hidden_dim) * scale  # candidate
        self.W_o = np.random.randn(concat_dim, hidden_dim) * scale  # output

        # Biases (forget gate bias initialized to 1 — remember by default)
        self.b_f = np.ones(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

    def forward(self, x_t, h_prev, c_prev):
        """One timestep. Returns new hidden state and cell state."""
        concat = np.concatenate([h_prev, x_t])      # (input_dim + hidden_dim,)

        # Gates (sigmoid squashes to [0, 1] — they're "soft switches")
        f = sigmoid(concat @ self.W_f + self.b_f)    # forget gate
        i = sigmoid(concat @ self.W_i + self.b_i)    # input gate
        c_hat = np.tanh(concat @ self.W_c + self.b_c) # candidate
        o = sigmoid(concat @ self.W_o + self.b_o)    # output gate

        # Cell state update: erase some old, add some new
        c_t = f * c_prev + i * c_hat                 # (hidden_dim,)

        # Hidden state: filtered view of cell state
        h_t = o * np.tanh(c_t)                       # (hidden_dim,)

        return h_t, c_t


# Smoke test
np.random.seed(42)
cell = LSTMCell(input_dim=3, hidden_dim=8)
h = np.zeros(8)
c = np.zeros(8)
for t in range(5):
    x = np.random.randn(3)
    h, c = cell.forward(x, h, c)
print(f"Final hidden state shape: {h.shape}")
print(f"Final cell state shape: {c.shape}")
print(f"Forget gate bias (should be 1s): {cell.b_f[:3]}")
print("LSTM cell test passed")
