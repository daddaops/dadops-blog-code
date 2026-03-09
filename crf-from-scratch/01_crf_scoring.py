import numpy as np

class LinearChainCRF:
    def __init__(self, n_labels, n_features):
        self.n_labels = n_labels
        # Emission weights: (n_labels, n_features)
        self.W = np.zeros((n_labels, n_features))
        # Transition weights: (n_labels, n_labels)
        # T[j, k] = score of transitioning from label j to label k
        self.T = np.zeros((n_labels, n_labels))

    def emission_scores(self, x):
        """Compute emission scores for each position and label."""
        # x: (seq_len, n_features), returns (seq_len, n_labels)
        return x @ self.W.T

    def score(self, x, y):
        """Score a complete label sequence y given input x."""
        emit = self.emission_scores(x)
        total = 0.0
        for i in range(len(y)):
            total += emit[i, y[i]]           # emission at position i
            if i > 0:
                total += self.T[y[i-1], y[i]] # transition from y[i-1] to y[i]
        return total

if __name__ == "__main__":
    # Example: 4 POS tags (0=DET, 1=NOUN, 2=VERB, 3=ADJ)
    tags = ["DET", "NOUN", "VERB", "ADJ"]
    crf = LinearChainCRF(n_labels=4, n_features=5)

    # Set transition weights to reflect English POS patterns
    crf.T = np.array([
        #  DET   NOUN  VERB  ADJ
        [-1.0,  2.0, -0.5,  1.5],  # after DET: NOUN or ADJ likely
        [-0.5, -0.5,  2.0,  0.0],  # after NOUN: VERB likely
        [ 1.0,  1.0, -1.0,  0.5],  # after VERB: DET or NOUN likely
        [-0.5,  2.0, -0.5, -1.0],  # after ADJ: NOUN likely
    ])

    # Dummy input: 4 words, 5 features each
    x = np.random.default_rng(42).standard_normal((4, 5))
    crf.W = np.random.default_rng(7).standard_normal((4, 5))

    seq1 = [0, 1, 2, 1]  # DET NOUN VERB NOUN (natural)
    seq2 = [0, 2, 2, 2]  # DET VERB VERB VERB (unnatural)

    print(f"Score of '{' '.join(tags[t] for t in seq1)}': {crf.score(x, seq1):.2f}")
    print(f"Score of '{' '.join(tags[t] for t in seq2)}': {crf.score(x, seq2):.2f}")
    # Output:
    # Score of 'DET NOUN VERB NOUN': 4.79
    # Score of 'DET VERB VERB VERB': -5.11
