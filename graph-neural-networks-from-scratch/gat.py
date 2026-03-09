import numpy as np
from graph_setup import A, X

def gat_layer(A, H, W, a, negative_slope=0.2):
    """
    Graph Attention Network layer (Velickovic et al. 2018).
    W: linear transform (F_in x F_out)
    a: attention vector (2*F_out,) - learns which neighbor pairs matter
    """
    N = H.shape[0]
    # Linear transform all nodes
    Z = H @ W  # (N x F_out)
    F_out = Z.shape[1]

    # Compute attention scores for all edges
    H_new = np.zeros_like(Z)

    for i in range(N):
        neighbors = np.where(A[i] > 0)[0]
        neighbors = np.append(neighbors, i)  # include self

        if len(neighbors) == 0:
            H_new[i] = Z[i]
            continue

        # Concatenate [z_i || z_j] for each neighbor j
        z_i_repeated = np.tile(Z[i], (len(neighbors), 1))  # (num_neigh x F_out)
        z_j = Z[neighbors]                                  # (num_neigh x F_out)
        concat = np.concatenate([z_i_repeated, z_j], axis=1)  # (num_neigh x 2*F_out)

        # Attention scores: e_ij = LeakyReLU(a^T . [z_i || z_j])
        e = concat @ a  # (num_neigh,)
        e = np.where(e > 0, e, e * negative_slope)  # LeakyReLU

        # Softmax over neighbors
        e_exp = np.exp(e - e.max())  # numerical stability
        alpha = e_exp / (e_exp.sum() + 1e-8)

        # Weighted aggregation
        H_new[i] = (alpha[:, None] * z_j).sum(axis=0)

    return np.maximum(0, H_new)  # ReLU

if __name__ == "__main__":
    np.random.seed(42)
    W_gat = np.random.randn(3, 4) * 0.5
    a_vec = np.random.randn(8) * 0.5  # 2 * F_out = 8

    H_gat = gat_layer(A, X, W_gat, a_vec)
    print("GAT output shape:", H_gat.shape)  # (6, 4)
