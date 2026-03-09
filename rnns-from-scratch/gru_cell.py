"""GRU cell — simpler than LSTM, often equally effective."""
import numpy as np
from helpers import sigmoid


class GRUCell:
    """A single GRU cell — simpler than LSTM, often equally effective."""
    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim
        scale = np.sqrt(2.0 / (concat_dim + hidden_dim))

        self.W_r = np.random.randn(concat_dim, hidden_dim) * scale  # reset
        self.W_z = np.random.randn(concat_dim, hidden_dim) * scale  # update
        self.W_h = np.random.randn(concat_dim, hidden_dim) * scale  # candidate

        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

    def forward(self, x_t, h_prev):
        """One timestep. Returns new hidden state."""
        concat = np.concatenate([h_prev, x_t])

        r = sigmoid(concat @ self.W_r + self.b_r)     # reset gate
        z = sigmoid(concat @ self.W_z + self.b_z)     # update gate

        # Candidate uses reset-gated version of previous hidden state
        concat_reset = np.concatenate([r * h_prev, x_t])
        h_hat = np.tanh(concat_reset @ self.W_h + self.b_h)

        # Interpolate between old and new
        h_t = (1 - z) * h_prev + z * h_hat            # (hidden_dim,)
        return h_t


# Smoke test
np.random.seed(42)
cell = GRUCell(input_dim=3, hidden_dim=8)
h = np.zeros(8)
for t in range(5):
    x = np.random.randn(3)
    h = cell.forward(x, h)
print(f"Final hidden state shape: {h.shape}")
print("GRU cell test passed")
