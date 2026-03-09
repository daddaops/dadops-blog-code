import numpy as np

# A simple 1D signal
signal = np.array([0, 0, 1, 3, 5, 3, 1, 0, 0, 0], dtype=float)

# A shift (translation) by 2 positions to the right
def shift(x, k):
    return np.roll(x, k)

shifted = shift(signal, 2)  # [0, 0, 0, 0, 1, 3, 5, 3, 1, 0]

# INVARIANT operation: sum
print(f"sum(signal)  = {signal.sum()}")   # 13.0
print(f"sum(shifted) = {shifted.sum()}")  # 13.0  -- same!

# EQUIVARIANT operation: convolution with a kernel
kernel = np.array([1, -1])  # edge detector
conv_then_shift = shift(np.convolve(signal, kernel, mode='same'), 2)
shift_then_conv = np.convolve(shifted, kernel, mode='same')

print(f"conv then shift: {conv_then_shift}")
print(f"shift then conv: {shift_then_conv}")
# They match! Convolution commutes with translation.
