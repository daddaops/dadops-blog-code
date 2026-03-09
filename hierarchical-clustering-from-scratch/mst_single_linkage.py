import numpy as np
from lance_williams import hierarchical_cluster

class UnionFind:
    """Disjoint-set with path compression and union by rank."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

def mst_single_linkage(X):
    """Single-linkage clustering via MST (Kruskal's algorithm)."""
    n = len(X)
    # All pairwise edges, sorted by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            edges.append((d, i, j))
    edges.sort()

    uf = UnionFind(n)
    linkage = []
    cluster_id = {i: i for i in range(n)}
    next_id = n

    for d, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri != rj:
            ci, cj = cluster_id[ri], cluster_id[rj]
            uf.union(ri, rj)
            new_root = uf.find(ri)
            new_size = uf.size[new_root]
            linkage.append([ci, cj, d, new_size])
            cluster_id[new_root] = next_id
            next_id += 1
        if len(linkage) == n - 1:
            break

    return np.array(linkage)

if __name__ == "__main__":
    # Verify equivalence
    np.random.seed(99)
    X = np.random.randn(8, 2)
    Z_agg = hierarchical_cluster(X, "single")
    Z_mst = mst_single_linkage(X)
    print("Agglomerative merge distances:", Z_agg[:, 2].round(3))
    print("MST merge distances:          ", Z_mst[:, 2].round(3))
    print("Match:", np.allclose(sorted(Z_agg[:, 2]), sorted(Z_mst[:, 2])))
