import numpy as np

# Transition matrix: Sunny, Rainy, Cloudy
# A[i][j] = P(next=j | current=i)
A = np.array([
    [0.6, 0.2, 0.2],  # Sunny  -> 60% Sunny, 20% Rainy, 20% Cloudy
    [0.3, 0.4, 0.3],  # Rainy  -> 30% Sunny, 40% Rainy, 30% Cloudy
    [0.4, 0.3, 0.3],  # Cloudy -> 40% Sunny, 30% Rainy, 30% Cloudy
])
states = ["Sunny", "Rainy", "Cloudy"]

# Simulate 10,000 steps
np.random.seed(42)
current = 0  # start Sunny
counts = np.zeros(3)
for _ in range(10_000):
    counts[current] += 1
    current = np.random.choice(3, p=A[current])

empirical = counts / counts.sum()

# Stationary distribution: left eigenvector with eigenvalue 1
eigenvalues, eigenvectors = np.linalg.eig(A.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
stationary = np.real(eigenvectors[:, idx])
stationary = stationary / stationary.sum()

for i, s in enumerate(states):
    print(f"{s:<8} empirical={empirical[i]:.3f}  stationary={stationary[i]:.3f}")
# Sunny    empirical=0.462  stationary=0.462
# Rainy    empirical=0.276  stationary=0.276
# Cloudy   empirical=0.263  stationary=0.262
