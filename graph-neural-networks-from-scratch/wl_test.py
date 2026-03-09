import numpy as np
from graph_setup import A

def wl_color_refinement(A, num_iters=3):
    """
    Weisfeiler-Leman color refinement (1-WL).
    Iteratively updates node colors based on neighbor color multisets.
    """
    N = A.shape[0]
    colors = list(range(N))  # start: each node has a unique color
    color_map = {}
    next_color = N

    for iteration in range(num_iters):
        new_colors = []
        for v in range(N):
            neighbors = np.where(A[v] > 0)[0]
            # Hash: (own color, sorted tuple of neighbor colors)
            neighbor_colors = tuple(sorted(colors[u] for u in neighbors))
            key = (colors[v], neighbor_colors)

            if key not in color_map:
                color_map[key] = next_color
                next_color += 1
            new_colors.append(color_map[key])

        colors = new_colors
        print(f"  Iter {iteration + 1}: {colors}")

    return colors

if __name__ == "__main__":
    print("WL on our graph:")
    wl_colors = wl_color_refinement(A, num_iters=3)
    # Nodes with identical local structure get the same final color
