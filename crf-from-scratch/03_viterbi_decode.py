import numpy as np

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

def viterbi_decode(emit, T):
    """Find the highest-scoring label sequence.

    Returns: best score and best label sequence
    """
    n, K = emit.shape
    delta = np.full((n, K), -np.inf)
    psi = np.zeros((n, K), dtype=int)  # backpointers

    # Base case
    delta[0] = emit[0]

    # Forward pass: max instead of logsumexp
    for i in range(1, n):
        for k in range(K):
            scores = delta[i-1] + T[:, k]  # score from each previous label
            psi[i, k] = np.argmax(scores)   # best previous label
            delta[i, k] = emit[i, k] + scores[psi[i, k]]

    # Traceback: follow backpointers from the end
    y_best = np.zeros(n, dtype=int)
    y_best[-1] = np.argmax(delta[-1])
    for i in range(n - 2, -1, -1):
        y_best[i] = psi[i + 1, y_best[i + 1]]

    return delta[-1].max(), y_best

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

    emit = crf.emission_scores(x)
    best_score, best_seq = viterbi_decode(emit, crf.T)
    greedy_seq = np.argmax(emit, axis=1)  # independent per-position argmax

    print(f"Viterbi: {' '.join(tags[t] for t in best_seq)} (score: {best_score:.2f})")
    print(f"Greedy:  {' '.join(tags[t] for t in greedy_seq)} (score: {crf.score(x, greedy_seq):.2f})")
    # Output:
    # Viterbi: ADJ NOUN VERB DET (score: 7.67)
    # Greedy:  ADJ NOUN VERB DET (score: 7.67)
