"""PCGrad: Projected Conflicting Gradients.

Detects and resolves gradient conflicts between tasks by projecting
out harmful components before summing gradients.
"""
import numpy as np


def pcgrad(task_gradients):
    """PCGrad: project conflicting gradients to remove harmful components.
    task_gradients: list of gradient vectors [g_1, g_2, ...]
    Returns: combined update direction (sum of modified gradients).
    """
    modified = [g.copy() for g in task_gradients]
    n_tasks = len(modified)
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                continue
            dot = np.dot(modified[i], task_gradients[j])
            if dot < 0:  # conflict detected
                # Project out the conflicting component
                norm_sq = np.dot(task_gradients[j], task_gradients[j])
                if norm_sq > 1e-12:
                    modified[i] -= (dot / norm_sq) * task_gradients[j]
    # Final update direction: sum of modified gradients
    return sum(modified)


if __name__ == "__main__":
    # Example: two 4D gradients that conflict
    g_reg = np.array([2.0, -1.0, 0.5, 3.0])    # regression wants to go "northeast"
    g_cls = np.array([-1.5, 0.8, 0.3, -2.0])   # classification wants to go "southwest"
    print(f"Cosine similarity: {np.dot(g_reg, g_cls) / (np.linalg.norm(g_reg) * np.linalg.norm(g_cls)):.3f}")
    combined = pcgrad([g_reg, g_cls])
    print(f"PCGrad direction: {combined}")

    # Show that PCGrad doesn't hurt aligned gradients
    g_aligned = np.array([1.0, 2.0, 3.0, 4.0])
    g_same = np.array([2.0, 1.0, 2.0, 3.0])
    cos_before = np.dot(g_aligned, g_same) / (np.linalg.norm(g_aligned) * np.linalg.norm(g_same))
    combined_aligned = pcgrad([g_aligned, g_same])
    print(f"\nAligned gradients cos={cos_before:.3f}")
    print(f"PCGrad (aligned): {combined_aligned}")
    print(f"Simple sum:       {g_aligned + g_same}")
    print("When aligned, PCGrad = simple sum (no projection needed)")
