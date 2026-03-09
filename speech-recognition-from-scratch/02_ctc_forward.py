"""CTC forward algorithm: compute CTC loss for a target sequence."""
import numpy as np


def ctc_forward(log_probs, target, blank=0):
    """CTC forward algorithm.
    log_probs: (T, C) log probabilities per frame per character
    target: list of character indices (without blanks)
    Returns: negative log-likelihood (CTC loss)
    """
    T = log_probs.shape[0]
    # Build expanded label: insert blanks between chars and at ends
    L = len(target)
    S = 2 * L + 1
    expanded = [blank] * S
    for i in range(L):
        expanded[2 * i + 1] = target[i]

    # Alpha matrix: log domain for numerical stability
    LOG_ZERO = -1e30
    alpha = np.full((T, S), LOG_ZERO)

    # Initialize: can start at blank or first char
    alpha[0, 0] = log_probs[0, expanded[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, expanded[1]]

    # Fill the trellis
    for t in range(1, T):
        for s in range(S):
            # Stay at same position
            a = alpha[t-1, s]
            # Move from previous position
            if s > 0:
                a = np.logaddexp(a, alpha[t-1, s-1])
            # Skip blank (char_i -> char_j, i != j)
            if s > 1 and expanded[s] != blank and expanded[s] != expanded[s-2]:
                a = np.logaddexp(a, alpha[t-1, s-2])
            alpha[t, s] = a + log_probs[t, expanded[s]]

    # Total probability: end at last blank or last char
    loss = np.logaddexp(alpha[T-1, S-1], alpha[T-1, S-2])
    return -loss, alpha


if __name__ == "__main__":
    # Example: target "cat" with T=10 frames, C=5 (blank=0, c=1, a=2, t=3, x=4)
    np.random.seed(7)
    T, C = 10, 5
    logits = np.random.randn(T, C) * 0.5
    # Bias frames toward correct alignment
    for t in range(0, 3): logits[t, 1] += 2.0   # c
    for t in range(3, 7): logits[t, 2] += 2.0   # a
    for t in range(7, 10): logits[t, 3] += 2.0  # t

    log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    target = [1, 2, 3]  # c, a, t

    loss, alpha = ctc_forward(log_probs, target)
    print(f"CTC loss (neg log-likelihood): {loss:.4f}")
    print(f"Alpha matrix shape: {alpha.shape}  (T={T} x S={2*len(target)+1})")
