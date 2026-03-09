"""Navigable Small World (NSW) Graph for ANN Search.

Builds a proximity graph incrementally where early-inserted points get
long-range connections and later points get short-range connections,
creating the small-world property for O(log n) search.
"""
import numpy as np
from heapq import heappush, heappop

def build_nsw(data, M=5, ef=20, seed=42):
    """Build a Navigable Small World graph."""
    rng = np.random.RandomState(seed)
    n = len(data)
    graph = {i: [] for i in range(n)}
    order = rng.permutation(n)

    def distance(a, b):
        diff = data[a] - data[b]
        return np.dot(diff, diff)

    def greedy_search(query_idx, entry, ef_search):
        """Beam search returning ef closest nodes."""
        visited = {entry}
        candidates = [(distance(query_idx, entry), entry)]
        results = [(-distance(query_idx, entry), entry)]

        while candidates:
            dist_c, c = heappop(candidates)
            worst_r = -results[0][0] if results else float('inf')
            if dist_c > worst_r and len(results) >= ef_search:
                break
            for nb in graph[c]:
                if nb not in visited:
                    visited.add(nb)
                    d = distance(query_idx, nb)
                    if len(results) < ef_search or d < -results[0][0]:
                        heappush(candidates, (d, nb))
                        heappush(results, (-d, nb))
                        if len(results) > ef_search:
                            heappop(results)
        return [idx for _, idx in results]

    # Insert points one by one
    for i, idx in enumerate(order):
        if i == 0:
            continue
        entry = order[rng.randint(0, i)]
        neighbors = greedy_search(idx, entry, ef)
        neighbors.sort(key=lambda nb: distance(idx, nb))
        for nb in neighbors[:M]:
            if nb not in graph[idx]:
                graph[idx].append(nb)
            if idx not in graph[nb]:
                graph[nb].append(idx)

    return graph, order[0]

def search_nsw(data, graph, entry, query, top_k=5, ef=30):
    """Search NSW graph for approximate nearest neighbors of a query vector."""
    visited = {entry}
    d_entry = np.sum((data[entry] - query) ** 2)
    candidates = [(d_entry, entry)]
    results = [(-d_entry, entry)]

    while candidates:
        dist_c, c = heappop(candidates)
        worst_r = -results[0][0] if results else float('inf')
        if dist_c > worst_r and len(results) >= ef:
            break
        for nb in graph[c]:
            if nb not in visited:
                visited.add(nb)
                d = np.sum((data[nb] - query) ** 2)
                if len(results) < ef or d < -results[0][0]:
                    heappush(candidates, (d, nb))
                    heappush(results, (-d, nb))
                    if len(results) > ef:
                        heappop(results)

    top = sorted([(np.sum((data[idx] - query)**2), idx) for _, idx in results])
    return [idx for _, idx in top[:top_k]]

if __name__ == "__main__":
    # Test on 2000 points in 32-d
    rng = np.random.RandomState(0)
    data = rng.randn(2000, 32)
    graph, entry = build_nsw(data, M=5, ef=20)

    query = rng.randn(32)
    dists = np.sum((data - query)**2, axis=1)
    true_top5 = set(np.argsort(dists)[:5])

    nsw_result = search_nsw(data, graph, entry, query, top_k=5, ef=30)
    recall = len(set(nsw_result) & true_top5) / 5
    print(f"NSW recall@5: {recall:.1f}")
