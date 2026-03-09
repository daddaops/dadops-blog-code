import numpy as np
from scipy.special import logsumexp
from itertools import product

class LinearChainCRF:
    def __init__(self, n_labels, n_features):
        self.n_labels = n_labels
        self.W = np.zeros((n_labels, n_features))
        self.T = np.zeros((n_labels, n_labels))

    def emission_scores(self, x):
        return x @ self.W.T

    def score(self, x, y):
        emit = self.emission_scores(x)
        total = 0.0
        for i in range(len(y)):
            total += emit[i, y[i]]
            if i > 0:
                total += self.T[y[i-1], y[i]]
        return total

def forward_algorithm(emit, T):
    """Compute log Z(x) using the forward algorithm.

    emit: (seq_len, n_labels) emission scores
    T:    (n_labels, n_labels) transition scores, T[j,k] = j -> k
    Returns: log Z(x) and forward table alpha
    """
    n, K = emit.shape
    alpha = np.full((n, K), -np.inf)

    # Base case: first position has only emission scores
    alpha[0] = emit[0]

    # Fill forward table
    for i in range(1, n):
        for k in range(K):
            # All paths arriving at (i, k): previous label j, transition j->k
            alpha[i, k] = emit[i, k] + logsumexp(alpha[i-1] + T[:, k])

    # Total log partition function
    log_Z = logsumexp(alpha[-1])
    return log_Z, alpha

if __name__ == "__main__":
    tags = ["DET", "NOUN", "VERB", "ADJ"]
    crf = LinearChainCRF(n_labels=4, n_features=5)

    crf.T = np.array([
        [-1.0,  2.0, -0.5,  1.5],
        [-0.5, -0.5,  2.0,  0.0],
        [ 1.0,  1.0, -1.0,  0.5],
        [-0.5,  2.0, -0.5, -1.0],
    ])

    x = np.random.default_rng(42).standard_normal((4, 5))
    crf.W = np.random.default_rng(7).standard_normal((4, 5))

    # Verify against brute force on our small example
    emit = crf.emission_scores(x)
    log_Z_forward, alpha = forward_algorithm(emit, crf.T)

    # Brute force: enumerate all 4^4 = 256 sequences
    all_scores = []
    for seq in product(range(4), repeat=4):
        all_scores.append(crf.score(x, list(seq)))
    log_Z_brute = logsumexp(all_scores)

    print(f"log Z (forward):     {log_Z_forward:.6f}")
    print(f"log Z (brute force): {log_Z_brute:.6f}")
    print(f"Match: {np.isclose(log_Z_forward, log_Z_brute)}")
    # Output:
    # log Z (forward):     9.512743
    # log Z (brute force): 9.512743
    # Match: True
