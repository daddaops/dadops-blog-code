import numpy as np
from graph_setup import A, X

def graphsage_layer(A, H, W_self, W_neigh, num_samples=3):
    """
    GraphSAGE layer with mean aggregator and neighbor sampling.
    Learns separate transforms for self-features and neighbor-features.
    """
    N = H.shape[0]
    H_new = np.zeros((N, W_self.shape[1] + W_neigh.shape[1]))

    for v in range(N):
        # Find neighbors
        neighbors = np.where(A[v] > 0)[0]

        if len(neighbors) == 0:
            agg = np.zeros(H.shape[1])
        else:
            # Sample fixed number of neighbors (with replacement if needed)
            sampled = np.random.choice(neighbors,
                        size=min(num_samples, len(neighbors)), replace=False)
            # Mean aggregation over sampled neighbors
            agg = H[sampled].mean(axis=0)

        # Concatenate self features with aggregated neighbor features
        combined = np.concatenate([H[v] @ W_self, agg @ W_neigh])

        # ReLU activation
        H_new[v] = np.maximum(0, combined)

    # L2 normalize (as in the original paper)
    norms = np.linalg.norm(H_new, axis=1, keepdims=True) + 1e-8
    return H_new / norms

if __name__ == "__main__":
    np.random.seed(42)
    W_self = np.random.randn(3, 4) * 0.5
    W_neigh = np.random.randn(3, 4) * 0.5

    H_sage = graphsage_layer(A, X, W_self, W_neigh, num_samples=2)
    print("GraphSAGE output shape:", H_sage.shape)  # (6, 8) - 4 self + 4 neighbor
