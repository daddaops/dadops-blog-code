import numpy as np

def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    return e / np.sum(e)

def continuous_hopfield_energy(patterns, x, beta=1.0):
    """Energy: -log sum exp(beta * xi . x) + 0.5 * x.x"""
    dots = np.array([beta * p @ x for p in patterns])
    lse = np.max(dots) + np.log(np.sum(np.exp(dots - np.max(dots))))
    return -lse + 0.5 * np.dot(x, x)

def continuous_hopfield_update(patterns, x, beta=1.0):
    """One update step: x_new = sum softmax(beta * xi . x) * xi"""
    dots = np.array([beta * p @ x for p in patterns])
    weights = softmax(dots)
    return sum(w * p for w, p in zip(weights, patterns))

if __name__ == "__main__":
    # Store 5 patterns in R^20, retrieve from noisy query
    np.random.seed(42)
    d = 20
    stored = [np.random.randn(d) for _ in range(5)]
    # Normalize patterns for cleaner retrieval
    stored = [p / np.linalg.norm(p) * np.sqrt(d) for p in stored]

    # Corrupt pattern 0 with substantial noise
    query = stored[0] + np.random.randn(d) * 1.5
    print(f"Initial distance to pattern 0: {np.linalg.norm(query - stored[0]):.3f}")

    # Iterate updates (lower beta = softer attention, shows gradual convergence)
    x = query.copy()
    for step in range(5):
        e = continuous_hopfield_energy(stored, x, beta=0.5)
        x = continuous_hopfield_update(stored, x, beta=0.5)
        dist = np.linalg.norm(x - stored[0])
        print(f"Step {step+1}: energy={e:.3f}, dist_to_p0={dist:.4f}")
    # Initial distance to pattern 0: 6.693
    # Step 1: energy=18.105, dist_to_p0=0.7075
    # Step 2: energy=-0.996, dist_to_p0=0.0197
    # Step 3: energy=-0.029, dist_to_p0=0.0063
    # Step 4: energy=-0.010, dist_to_p0=0.0061
    # Step 5: energy=-0.009, dist_to_p0=0.0061
