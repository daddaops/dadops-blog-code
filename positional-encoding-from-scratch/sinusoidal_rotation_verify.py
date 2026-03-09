"""Verify sinusoidal encoding's linear transformability property.

PE(pos+k) = rotation_matrix(k) @ PE(pos) — relative position
can be expressed as a linear transformation of absolute position.
"""
import numpy as np

def sinusoidal_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)
    return pe

# Verify: PE(pos+k) = rotation_matrix(k) @ PE(pos)
pe = sinusoidal_encoding(20, 8)
omega = 1.0 / 10000 ** (0 / 8)  # Frequency for dims 0,1 (omega = 1.0)

k = 3  # Offset of 3 positions
# Rotation matrix for offset k at this frequency
M_k = np.array([
    [ np.cos(k * omega),  np.sin(k * omega)],
    [-np.sin(k * omega),  np.cos(k * omega)]
])

# Take PE at position 4, dims [0,1]
pe_4 = pe[4, :2]           # [sin(4), cos(4)]
pe_7_actual = pe[7, :2]    # [sin(7), cos(7)]
pe_7_computed = M_k @ pe_4 # Should equal pe[7, :2]

print(f"PE(7) actual:   {np.round(pe_7_actual, 6)}")
print(f"PE(4) x M_3:    {np.round(pe_7_computed, 6)}")
print(f"Match: {np.allclose(pe_7_actual, pe_7_computed)}")
