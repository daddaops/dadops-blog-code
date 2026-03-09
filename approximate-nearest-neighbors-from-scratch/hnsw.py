"""Hierarchical Navigable Small World (HNSW) Graph for ANN Search.

Multi-layer graph inspired by skip lists. Upper layers have few nodes with
long-range connections for coarse navigation. Layer 0 has all nodes for
precise search. Achieves O(log n) query complexity.
"""
import numpy as np
from heapq import heappush, heappop
import math

def build_hnsw(data, M=5, ef=20, mL=1.0, seed=42):
    """Build an HNSW index with multi-layer skip-list structure."""
    rng = np.random.RandomState(seed)
    n = len(data)

    # Assign layers: P(layer >= l) = exp(-l / mL)
    max_layers = [int(-math.log(rng.random() + 1e-10) * mL) for _ in range(n)]
    top_layer = max(max_layers)

    # One graph per layer
    layers = [{} for _ in range(top_layer + 1)]
    for i in range(n):
        for l in range(max_layers[i] + 1):
            layers[l][i] = []

    def distance(a, b):
        diff = data[a] - data[b]
        return np.dot(diff, diff)

    def search_layer(query_idx, entry, ef_search, layer):
        """Beam search within a single layer."""
        visited = {entry}
        d_e = distance(query_idx, entry)
        candidates = [(d_e, entry)]
        results = [(-d_e, entry)]
        while candidates:
            dist_c, c = heappop(candidates)
            if dist_c > -results[0][0] and len(results) >= ef_search:
                break
            for nb in layers[layer].get(c, []):
                if nb not in visited:
                    visited.add(nb)
                    d = distance(query_idx, nb)
                    if len(results) < ef_search or d < -results[0][0]:
                        heappush(candidates, (d, nb))
                        heappush(results, (-d, nb))
                        if len(results) > ef_search:
                            heappop(results)
        return [idx for _, idx in results]

    entry_point = 0
    order = rng.permutation(n)
    for idx in order:
        ep = entry_point
        # Descend from top to node's layer + 1 (greedy, ef=1)
        for l in range(top_layer, max_layers[idx] + 1, -1):
            if ep in layers[l]:
                nbs = search_layer(idx, ep, 1, l)
                ep = nbs[0]
        # Search and connect at each of node's layers
        for l in range(min(max_layers[idx], top_layer), -1, -1):
            if ep not in layers[l]:
                continue
            nbs = search_layer(idx, ep, ef, l)
            nbs.sort(key=lambda nb: distance(idx, nb))
            for nb in nbs[:M]:
                if nb != idx:
                    if nb not in layers[l].get(idx, []):
                        layers[l].setdefault(idx, []).append(nb)
                    if idx not in layers[l].get(nb, []):
                        layers[l].setdefault(nb, []).append(idx)
            ep = nbs[0] if nbs else ep
        if max_layers[idx] > max_layers[entry_point]:
            entry_point = idx

    # Report layer sizes
    for l in range(top_layer + 1):
        print(f"  Layer {l}: {len(layers[l])} nodes")

    return layers, entry_point, top_layer

if __name__ == "__main__":
    # Build HNSW on 2000 points
    rng = np.random.RandomState(0)
    data = rng.randn(2000, 32)
    layers, ep, top = build_hnsw(data, M=5, ef=20, mL=1.0)
