"""Sinusoidal positional encoding (Vaswani et al. 2017).

Each position gets a unique vector using sin/cos at
geometrically spaced frequencies.
"""
import numpy as np

def sinusoidal_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]        # (seq_len, 1)

    # Frequencies: 1, 1/10000^(2/d), 1/10000^(4/d), ...
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)  # (d_model/2,)

    pe[:, 0::2] = np.sin(position / div_term)  # Even dims get sin
    pe[:, 1::2] = np.cos(position / div_term)  # Odd dims get cos
    return pe

pe = sinusoidal_encoding(10, 8)
print("Position 0:", np.round(pe[0], 3))
print("Position 1:", np.round(pe[1], 3))
print("Position 9:", np.round(pe[9], 3))
