"""Vanilla RNN: RNNCell + VanillaRNN with forward pass test."""
import numpy as np

class RNNCell:
    """A single vanilla RNN cell."""
    def __init__(self, input_dim, hidden_dim):
        # Xavier initialization
        scale_xh = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        self.W_xh = np.random.randn(input_dim, hidden_dim) * scale_xh
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale_hh
        self.b_h = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_t, h_prev):
        """One timestep: takes input x_t and previous hidden state h_prev."""
        # h_t = tanh(x_t @ W_xh + h_prev @ W_hh + b_h)
        self.x_t = x_t                             # (input_dim,)
        self.h_prev = h_prev                        # (hidden_dim,)
        self.z = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h  # (hidden_dim,)
        self.h_t = np.tanh(self.z)                  # (hidden_dim,)
        return self.h_t

class VanillaRNN:
    """RNN that processes a full sequence."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.cell = RNNCell(input_dim, hidden_dim)
        # Output projection: hidden state -> output
        scale_out = np.sqrt(2.0 / (hidden_dim + output_dim))
        self.W_out = np.random.randn(hidden_dim, output_dim) * scale_out
        self.b_out = np.zeros(output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, xs, h_init=None):
        """Process a sequence of inputs.
        xs: list of input vectors, one per timestep
        Returns: list of outputs and list of hidden states
        """
        if h_init is None:
            h_init = np.zeros(self.hidden_dim)

        hiddens = []
        outputs = []
        h = h_init

        for x_t in xs:
            h = self.cell.forward(x_t, h)           # (hidden_dim,)
            hiddens.append(h)
            y_t = h @ self.W_out + self.b_out        # (output_dim,)
            outputs.append(y_t)

        return outputs, hiddens

# Quick test: 5-step sequence, input_dim=3, hidden_dim=8, output_dim=4
rnn = VanillaRNN(input_dim=3, hidden_dim=8, output_dim=4)
xs = [np.random.randn(3) for _ in range(5)]
outputs, hiddens = rnn.forward(xs)

print(f"Sequence length: {len(xs)}")       # 5
print(f"Hidden state shape: {hiddens[0].shape}")  # (8,)
print(f"Output shape: {outputs[0].shape}")        # (4,)
print(f"Final hidden state:\n{hiddens[-1].round(3)}")
