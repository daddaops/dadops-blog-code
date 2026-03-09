"""Dot Product as Similarity Measure.

Shows how dot products between word embeddings measure similarity,
which is the foundation of the attention mechanism.
"""
import numpy as np

if __name__ == "__main__":
    # Two word embeddings (dimension 4 for simplicity)
    cat  = np.array([0.9, 0.1, 0.8, 0.2])   # "cat" — animal-like
    sat  = np.array([0.1, 0.9, 0.3, 0.7])   # "sat" — action-like
    mat  = np.array([0.8, 0.2, 0.7, 0.3])   # "mat" — object, similar to cat

    # Dot product measures similarity
    print(f"cat · sat = {np.dot(cat, sat):.3f}")   # moderate
    print(f"cat · mat = {np.dot(cat, mat):.3f}")   # high!
    print(f"sat · mat = {np.dot(sat, mat):.3f}")   # moderate
