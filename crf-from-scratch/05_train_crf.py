import numpy as np
from scipy.special import logsumexp

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

def forward_backward(emit, T):
    """Compute node and edge marginals via forward-backward."""
    n, K = emit.shape

    alpha = np.full((n, K), -np.inf)
    alpha[0] = emit[0]
    for i in range(1, n):
        for k in range(K):
            alpha[i, k] = emit[i, k] + logsumexp(alpha[i-1] + T[:, k])
    log_Z = logsumexp(alpha[-1])

    beta = np.full((n, K), -np.inf)
    beta[-1] = 0.0
    for i in range(n - 2, -1, -1):
        for j in range(K):
            beta[i, j] = logsumexp(T[j, :] + emit[i+1] + beta[i+1])

    node_marginals = np.exp(alpha + beta - log_Z)

    edge_marginals = np.zeros((n - 1, K, K))
    for i in range(1, n):
        for j in range(K):
            for k in range(K):
                edge_marginals[i-1, j, k] = np.exp(
                    alpha[i-1, j] + T[j, k] + emit[i, k] + beta[i, k] - log_Z
                )

    return node_marginals, edge_marginals, log_Z

def viterbi_decode(emit, T):
    """Find the highest-scoring label sequence."""
    n, K = emit.shape
    delta = np.full((n, K), -np.inf)
    psi = np.zeros((n, K), dtype=int)

    delta[0] = emit[0]
    for i in range(1, n):
        for k in range(K):
            scores = delta[i-1] + T[:, k]
            psi[i, k] = np.argmax(scores)
            delta[i, k] = emit[i, k] + scores[psi[i, k]]

    y_best = np.zeros(n, dtype=int)
    y_best[-1] = np.argmax(delta[-1])
    for i in range(n - 2, -1, -1):
        y_best[i] = psi[i + 1, y_best[i + 1]]

    return delta[-1].max(), y_best

def train_crf(X_train, Y_train, n_labels, n_features, lr=0.1, epochs=50, reg=0.01):
    """Train a CRF with SGD on labeled sequences."""
    crf = LinearChainCRF(n_labels, n_features)

    for epoch in range(epochs):
        total_ll = 0.0
        for x, y in zip(X_train, Y_train):
            emit = crf.emission_scores(x)
            node_marg, edge_marg, log_Z = forward_backward(emit, crf.T)
            score = crf.score(x, y)
            total_ll += score - log_Z

            grad_W = np.zeros_like(crf.W)
            for i in range(len(y)):
                grad_W[y[i]] += x[i]
                for k in range(n_labels):
                    grad_W[k] -= node_marg[i, k] * x[i]

            grad_T = np.zeros_like(crf.T)
            for i in range(1, len(y)):
                grad_T[y[i-1], y[i]] += 1.0
            grad_T -= edge_marg.sum(axis=0)

            crf.W += lr * (grad_W - reg * crf.W)
            crf.T += lr * (grad_T - reg * crf.T)

        if (epoch + 1) % 10 == 0:
            avg_ll = total_ll / len(X_train)
            print(f"Epoch {epoch+1}: avg log-likelihood = {avg_ll:.3f}")

    return crf

if __name__ == "__main__":
    tags = ["DET", "NOUN", "VERB", "ADJ"]

    # Train CRF vs independent classifier on synthetic POS data
    rng = np.random.default_rng(42)
    vocab = {"the": 0, "a": 1, "dog": 2, "cat": 3, "sat": 4, "ran": 5,
             "big": 6, "red": 7, "on": 8, "mat": 9}
    # Tag patterns: DET (ADJ) NOUN VERB PREP DET NOUN
    templates = [
        ([0, 2, 4, 8, 0, 9], [0, 1, 2, 0, 0, 1]),  # the dog sat on the mat
        ([1, 3, 5, 8, 0, 9], [0, 1, 2, 0, 0, 1]),  # a cat ran on the mat
        ([0, 6, 2, 4, 8, 1, 9], [0, 3, 1, 2, 0, 0, 1]),  # the big dog sat on a mat
        ([0, 7, 3, 5, 8, 0, 9], [0, 3, 1, 2, 0, 0, 1]),  # the red cat ran on the mat
    ]

    # Generate 40 training sequences with slight feature noise
    X_train, Y_train = [], []
    for _ in range(10):
        for words, labels in templates:
            n_feat = len(vocab)
            x = np.zeros((len(words), n_feat))
            for i, w in enumerate(words):
                x[i, w] = 1.0
                x[i] += rng.normal(0, 0.05, n_feat)  # add noise
            X_train.append(x)
            Y_train.append(labels)

    trained_crf = train_crf(X_train, Y_train, n_labels=4, n_features=len(vocab),
                            lr=0.05, epochs=50, reg=0.01)

    # Check learned transition matrix
    print("\nLearned transition weights (rows=from, cols=to):")
    print("       DET   NOUN  VERB  ADJ")
    for i, name in enumerate(["DET ", "NOUN", "VERB", "ADJ "]):
        print(f"  {name} [{', '.join(f'{v:+.1f}' for v in trained_crf.T[i])}]")

    # Test: decode a new sequence
    x_test = np.zeros((6, len(vocab)))
    for i, w in enumerate([0, 3, 4, 8, 1, 9]):  # the cat sat on a mat
        x_test[i, w] = 1.0
    _, pred = viterbi_decode(trained_crf.emission_scores(x_test), trained_crf.T)
    print(f"\nPrediction: {' '.join(tags[t] for t in pred)}")
    print(f"Expected:   DET NOUN VERB DET DET NOUN")
