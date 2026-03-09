import numpy as np
from lance_williams import hierarchical_cluster

def get_leaf_order(Z, n):
    """Compute leaf ordering from linkage matrix (no crossing branches)."""
    order = []
    def traverse(node_id):
        if node_id < n:
            order.append(node_id)
        else:
            row = Z[int(node_id) - n]
            traverse(int(row[0]))
            traverse(int(row[1]))
    traverse(n + len(Z) - 1)  # start from root
    return order

def cophenetic_corr(X, Z):
    """Correlation between original and cophenetic distances."""
    n = len(X)
    orig, coph = [], []
    # Build cluster membership timeline
    members = {i: {i} for i in range(n)}
    merge_dist = {}
    for i, (c1, c2, d, _) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        new_id = n + i
        for a in members[c1]:
            for b in members[c2]:
                merge_dist[(min(a, b), max(a, b))] = d
        members[new_id] = members[c1] | members[c2]
    for i in range(n):
        for j in range(i + 1, n):
            orig.append(np.sqrt(np.sum((X[i] - X[j]) ** 2)))
            coph.append(merge_dist[(i, j)])
    return np.corrcoef(orig, coph)[0, 1]

if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([np.random.randn(10, 2) + [0, 0],
                   np.random.randn(10, 2) + [5, 5],
                   np.random.randn(10, 2) + [10, 0]])
    Z = hierarchical_cluster(X, "ward")
    r = cophenetic_corr(X, Z)
    print(f"Ward's dendrogram cophenetic r = {r:.3f}")
