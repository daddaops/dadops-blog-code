"""Character-level RNN training infrastructure."""
import numpy as np
from helpers import sigmoid, softmax
from lstm_cell import LSTMCell
from gru_cell import GRUCell
from vanilla_rnn import RNNCell


def char_rnn_forward(cell, xs_onehot, W_out, b_out, h_init, c_init=None):
    """Forward pass for character-level language model.
    Returns outputs (logits), hidden states, and cell states (if LSTM).
    """
    hiddens, cells, outputs = [], [], []
    h = h_init
    c = c_init  # Only used for LSTM

    for x_t in xs_onehot:
        if c is not None:  # LSTM
            h, c = cell.forward(x_t, h, c)
            cells.append(c)
        else:  # Vanilla RNN or GRU
            h = cell.forward(x_t, h)

        hiddens.append(h)
        logits = h @ W_out + b_out                   # (vocab_size,)
        outputs.append(logits)

    return outputs, hiddens, cells


def cross_entropy_loss(logits_list, targets_onehot):
    """Average cross-entropy loss over all timesteps."""
    loss = 0.0
    for logits, target in zip(logits_list, targets_onehot):
        probs = softmax(logits)
        loss -= np.log(probs[target.argmax()] + 1e-12)
    return loss / len(logits_list)


# Smoke test: forward pass with all three cell types
np.random.seed(42)
input_dim = 26  # alphabet
hidden_dim = 32
vocab_size = 26

# Random one-hot sequence "hello" -> indices
seq = [7, 4, 11, 11, 14]  # h, e, l, l, o
xs = [np.eye(input_dim)[i] for i in seq[:-1]]
targets = [np.eye(vocab_size)[i] for i in seq[1:]]

for name, cell_cls, needs_c in [("Vanilla RNN", RNNCell, False),
                                  ("LSTM", LSTMCell, True),
                                  ("GRU", GRUCell, False)]:
    np.random.seed(42)
    cell = cell_cls(input_dim, hidden_dim)
    W_out = np.random.randn(hidden_dim, vocab_size) * 0.1
    b_out = np.zeros(vocab_size)
    h_init = np.zeros(hidden_dim)
    c_init = np.zeros(hidden_dim) if needs_c else None

    outputs, hiddens, cells = char_rnn_forward(cell, xs, W_out, b_out, h_init, c_init)
    loss = cross_entropy_loss(outputs, targets)
    print(f"{name:12s}: loss = {loss:.3f} (random baseline: {np.log(vocab_size):.3f})")
