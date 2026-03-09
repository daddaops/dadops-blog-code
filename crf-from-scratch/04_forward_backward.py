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

    # Forward pass (already implemented above)
    alpha = np.full((n, K), -np.inf)
    alpha[0] = emit[0]
    for i in range(1, n):
        for k in range(K):
            alpha[i, k] = emit[i, k] + logsumexp(alpha[i-1] + T[:, k])
    log_Z = logsumexp(alpha[-1])

    # Backward pass
    beta = np.full((n, K), -np.inf)
    beta[-1] = 0.0  # log(1) = 0
    for i in range(n - 2, -1, -1):
        for j in range(K):
            beta[i, j] = logsumexp(T[j, :] + emit[i+1] + beta[i+1])

    # Node marginals: P(y_i = k | x)
    node_marginals = np.exp(alpha + beta - log_Z)

    # Edge marginals: P(y_i = k, y_{i-1} = j | x)
    edge_marginals = np.zeros((n - 1, K, K))
    for i in range(1, n):
        for j in range(K):
            for k in range(K):
                edge_marginals[i-1, j, k] = np.exp(
                    alpha[i-1, j] + T[j, k] + emit[i, k] + beta[i, k] - log_Z
                )

    return node_marginals, edge_marginals, log_Z

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

            # Gradient for emission weights
            grad_W = np.zeros_like(crf.W)
            for i in range(len(y)):
                grad_W[y[i]] += x[i]                  # observed
                for k in range(n_labels):
                    grad_W[k] -= node_marg[i, k] * x[i]  # expected

            # Gradient for transition weights
            grad_T = np.zeros_like(crf.T)
            for i in range(1, len(y)):
                grad_T[y[i-1], y[i]] += 1.0           # observed
            grad_T -= edge_marg.sum(axis=0)            # expected

            # SGD update with L2 regularization
            crf.W += lr * (grad_W - reg * crf.W)
            crf.T += lr * (grad_T - reg * crf.T)

        if (epoch + 1) % 10 == 0:
            avg_ll = total_ll / len(X_train)
            print(f"Epoch {epoch+1}: avg log-likelihood = {avg_ll:.3f}")

    return crf

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
    node_marg, edge_marg, log_Z = forward_backward(emit, crf.T)

    print(f"log Z = {log_Z:.6f}")
    print(f"\nNode marginals (rows=positions, cols=labels):")
    print(f"       {'  '.join(f'{t:>5s}' for t in tags)}")
    for i in range(node_marg.shape[0]):
        print(f"  pos {i}: {' '.join(f'{v:.3f}' for v in node_marg[i])}")
    print(f"\nNode marginals sum per position: {node_marg.sum(axis=1)}")
    print(f"Edge marginals sum per edge: {edge_marg.sum(axis=(1,2))}")
