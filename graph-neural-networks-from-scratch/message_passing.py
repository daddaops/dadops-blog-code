import numpy as np
from graph_setup import A, X

def message_passing_layer(A, H, W, activation=True):
    """
    One round of message passing.
    A: adjacency matrix (N x N)
    H: node features (N x F_in)
    W: weight matrix (F_in x F_out)
    """
    # Step 1-2: Gather and aggregate neighbor features
    # A @ H: for each node, sum up the feature vectors of its neighbors
    messages = A @ H  # (N x F_in)

    # Step 3: Transform with learnable weights + nonlinearity
    H_new = messages @ W  # (N x F_out)

    if activation:
        H_new = np.maximum(0, H_new)  # ReLU

    return H_new

if __name__ == "__main__":
    # Initialize a random weight matrix
    np.random.seed(42)
    W = np.random.randn(3, 4) * 0.5  # 3 input features -> 4 output features

    H1 = message_passing_layer(A, X, W)
    print("After 1 layer, node 0's features:", H1[0].round(3))
    # Node 0 now encodes information from nodes 1 and 2 (its neighbors)
