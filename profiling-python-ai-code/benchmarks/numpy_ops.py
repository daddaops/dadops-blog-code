"""NumPy linear algebra workload: matrix operations."""
import numpy as np

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Matrix multiply
    A = rng.standard_normal((1000, 1000))
    B = rng.standard_normal((1000, 1000))
    C = A @ B

    # SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Eigendecomposition
    sym = A @ A.T
    eigenvalues, eigenvectors = np.linalg.eigh(sym)

    # Solve linear system
    b = rng.standard_normal(1000)
    x = np.linalg.solve(sym, b)
