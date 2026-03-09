import numpy as np
from graph_setup import A, X

def gcn_layer(A, H, W):
    """
    Graph Convolutional Network layer (Kipf & Welling 2017).
    The whole thing is just: sigma(D_inv_sqrt @ A_hat @ D_inv_sqrt @ H @ W)
    """
    N = A.shape[0]

    # Add self-loops: every node is its own neighbor
    A_hat = A + np.eye(N)

    # Compute degree matrix of A_hat
    D_hat = np.diag(A_hat.sum(axis=1))

    # Symmetric normalization: D^(-1/2) A D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

    # Propagate: normalized message passing + learned transform + ReLU
    H_new = np.maximum(0, A_norm @ H @ W)
    return H_new

if __name__ == "__main__":
    np.random.seed(42)

    # Two-layer GCN
    W1 = np.random.randn(3, 8) * 0.5   # 3 -> 8 features
    W2 = np.random.randn(8, 2) * 0.5   # 8 -> 2 features (for visualization)

    H1 = gcn_layer(A, X, W1)           # Layer 1: mix neighbors
    H2 = gcn_layer(A, H1, W2)          # Layer 2: mix neighbors-of-neighbors

    print("2-layer GCN output:")
    for i in range(6):
        print(f"  Node {i}: [{H2[i][0]:.3f}, {H2[i][1]:.3f}]")
