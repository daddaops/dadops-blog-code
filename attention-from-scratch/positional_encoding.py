"""Sinusoidal Positional Encoding from Scratch.

Generates position-dependent vectors using sine and cosine at different
frequencies, allowing the model to learn relative positions.
"""
import numpy as np

def positional_encoding(seq_len, d_model):
    """
    Generates sinusoidal positional encodings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]          # (seq_len, 1)
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)  # (d_model/2,)

    pe[:, 0::2] = np.sin(position / div_term)  # even dimensions
    pe[:, 1::2] = np.cos(position / div_term)  # odd dimensions

    return pe

if __name__ == "__main__":
    pe = positional_encoding(20, 64)
    print(f"Shape: {pe.shape}")      # (20, 64)
    print(f"PE[0][:6]: {pe[0, :6]}")  # Position 0
    print(f"PE[1][:6]: {pe[1, :6]}")  # Position 1 — different!
    print(f"PE[0] · PE[1] = {np.dot(pe[0], pe[1]):.3f}")  # High similarity (nearby)
    print(f"PE[0] · PE[19] = {np.dot(pe[0], pe[19]):.3f}") # Lower (far apart)
