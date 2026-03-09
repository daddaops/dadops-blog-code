import numpy as np
from graph_setup import A, X
from gcn import gcn_layer

def graph_readout(H, mode='mean'):
    """
    Collapse node features into a single graph-level vector.
    H: node features (N x F)
    mode: 'sum', 'mean', or 'max'
    """
    if mode == 'sum':
        return H.sum(axis=0)     # preserves graph size info
    elif mode == 'mean':
        return H.mean(axis=0)    # size-invariant
    elif mode == 'max':
        return H.max(axis=0)     # captures extremes
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    np.random.seed(42)
    # After running a GCN, pool nodes to get a graph embedding
    H_final = gcn_layer(A, X, np.random.randn(3, 4) * 0.5)
    graph_vec = graph_readout(H_final, mode='mean')
    print("Graph vector:", graph_vec.round(3))  # One vector for the entire graph
    # Feed this into a classifier for graph-level prediction
