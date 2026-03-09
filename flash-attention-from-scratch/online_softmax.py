import numpy as np

def softmax_standard(x):
    """Standard 3-pass softmax."""
    m = np.max(x)                         # Pass 1: find max
    e = np.exp(x - m)                     # Pass 2: subtract max, exponentiate
    return e / np.sum(e)                  # Pass 3: normalize

def softmax_online(x, block_size=3):
    """Online softmax: processes values in blocks, maintaining running stats."""
    N = len(x)
    m = -np.inf   # running maximum
    l = 0.0       # running sum of exp(x_i - m)

    # First pass: compute global max and sum in blocks
    for start in range(0, N, block_size):
        block = x[start : start + block_size]
        m_block = np.max(block)

        # If this block has a new max, rescale previous sum
        m_new = max(m, m_block)
        l = l * np.exp(m - m_new) + np.sum(np.exp(block - m_new))
        m = m_new

    # Second pass: compute final softmax values
    return np.exp(x - m) / l

if __name__ == "__main__":
    # Demonstrate equivalence
    x = np.array([1.0, 2.0, 3.0, 6.0, 2.0, 1.0])
    standard = softmax_standard(x)
    online = softmax_online(x, block_size=3)

    print("Input:", x)
    print("Standard softmax:", np.round(standard, 6))
    print("Online softmax:  ", np.round(online, 6))
    print("Max difference:  ", np.max(np.abs(standard - online)))
