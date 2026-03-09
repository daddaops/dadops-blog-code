import numpy as np

def lance_williams_params(linkage, n_a, n_b, n_c):
    """Return (alpha_a, alpha_b, beta, gamma) for each linkage."""
    if linkage == "single":
        return 0.5, 0.5, 0, -0.5
    elif linkage == "complete":
        return 0.5, 0.5, 0, 0.5
    elif linkage == "average":
        return n_a / (n_a + n_b), n_b / (n_a + n_b), 0, 0
    elif linkage == "ward":
        n_t = n_a + n_b + n_c
        return (n_a + n_c) / n_t, (n_b + n_c) / n_t, -n_c / n_t, 0

def hierarchical_cluster(X, linkage="ward"):
    """Agglomerative clustering with configurable linkage."""
    n = len(X)
    dist = np.full((2 * n, 2 * n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            if linkage == "ward":
                d = np.sum((X[i] - X[j]) ** 2)  # squared Euclidean
            else:
                d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            dist[i, j] = dist[j, i] = d

    active = set(range(n))
    sizes = {i: 1 for i in range(n)}
    result = []

    for step in range(n - 1):
        min_d, ci, cj = np.inf, -1, -1
        for i in active:
            for j in active:
                if i < j and dist[i, j] < min_d:
                    min_d, ci, cj = dist[i, j], i, j

        new_id = n + step
        new_size = sizes[ci] + sizes[cj]
        merge_dist = np.sqrt(min_d) if linkage == "ward" else min_d
        result.append([ci, cj, merge_dist, new_size])
        sizes[new_id] = new_size

        for k in active:
            if k != ci and k != cj:
                a_a, a_b, b, g = lance_williams_params(
                    linkage, sizes[ci], sizes[cj], sizes[k])
                dist[new_id, k] = (a_a * dist[ci, k] + a_b * dist[cj, k]
                                   + b * dist[ci, cj]
                                   + g * abs(dist[ci, k] - dist[cj, k]))
                dist[k, new_id] = dist[new_id, k]

        active.discard(ci)
        active.discard(cj)
        active.add(new_id)

    return np.array(result)

if __name__ == "__main__":
    # Compare linkages on the same data
    X = np.array([[0, 0], [0.5, 0], [3, 0], [3.5, 0],
                  [6, 0], [6.3, 0], [6.6, 0]])
    for method in ["single", "complete", "average", "ward"]:
        Z = hierarchical_cluster(X, method)
        print(f"\n{method.upper()} linkage:")
        for row in Z:
            print(f"  {int(row[0]):2d}+{int(row[1]):2d} dist={row[2]:.3f}")
