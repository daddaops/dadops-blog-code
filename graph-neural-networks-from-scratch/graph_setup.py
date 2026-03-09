import numpy as np

# A small social network: 6 people
# Edges: 0-1, 0-2, 1-2, 1-3, 3-4, 3-5, 4-5
A = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
])

# Each person has a 3-dimensional feature vector
X = np.array([
    [1.0, 0.0, 0.2],  # Node 0
    [0.8, 0.3, 0.1],  # Node 1
    [0.9, 0.1, 0.3],  # Node 2
    [0.1, 0.8, 0.5],  # Node 3
    [0.0, 0.9, 0.7],  # Node 4
    [0.2, 0.7, 0.6],  # Node 5
])

# Degree matrix: how many connections each node has
D = np.diag(A.sum(axis=1))  # D[i,i] = number of neighbors of node i

if __name__ == "__main__":
    print("Degrees:", A.sum(axis=1))  # [2, 3, 2, 3, 2, 2]
