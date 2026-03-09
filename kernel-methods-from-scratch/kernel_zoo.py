import numpy as np

def k_linear(x, y):
    return np.dot(x, y)

def k_poly(x, y, degree=3, c=1.0):
    return (np.dot(x, y) + c) ** degree

def k_rbf(x, y, sigma=1.0):
    return np.exp(-np.sum((x - y)**2) / (2 * sigma**2))

def k_laplacian(x, y, sigma=1.0):
    return np.exp(-np.sum(np.abs(x - y)) / sigma)

def k_matern32(x, y, sigma=1.0):
    r = np.sqrt(np.sum((x - y)**2)) / sigma
    return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)

def k_cosine(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return np.dot(x, y) / (nx * ny)

# Visualize: fix reference point, compute kernel over a grid
x0 = np.array([1.0, 0.0])
grid = np.linspace(-3, 3, 100)
xx, yy = np.meshgrid(grid, grid)

kernels = {"Linear": k_linear, "Poly(3)": k_poly, "RBF": k_rbf,
           "Laplacian": k_laplacian, "Matern 3/2": k_matern32,
           "Cosine": k_cosine}

for name, kfn in kernels.items():
    heatmap = np.zeros_like(xx)
    for i in range(100):
        for j in range(100):
            heatmap[i, j] = kfn(x0, np.array([xx[i, j], yy[i, j]]))
    print(f"{name:<12} range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
