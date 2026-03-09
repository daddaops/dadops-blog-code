"""Monge's optimal transport via the Hungarian algorithm.

Finds the optimal 1-to-1 assignment between source and target points
minimizing total squared Euclidean distance.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# Source and target points in 2D
source = np.array([[1, 2], [3, 1], [2, 4], [5, 3], [4, 5]])
target = np.array([[6, 2], [8, 4], [7, 1], [9, 5], [5, 6]])

# Cost matrix: squared Euclidean distance between all pairs
n = len(source)
cost_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        cost_matrix[i, j] = np.sum((source[i] - target[j]) ** 2)

# Solve the assignment problem (Monge's optimal transport)
row_ind, col_ind = linear_sum_assignment(cost_matrix)

total_cost = cost_matrix[row_ind, col_ind].sum()
print(f"Optimal assignment cost: {total_cost:.2f}")
for i, j in zip(row_ind, col_ind):
    print(f"  Source {source[i]} -> Target {target[j]}"
          f"  (cost: {cost_matrix[i, j]:.1f})")
