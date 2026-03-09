import numpy as np

def rbf_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
    """Radial Basis Function (squared exponential) kernel."""
    sq_dist = np.sum(X1**2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)
    return signal_var * np.exp(-0.5 * sq_dist / length_scale**2)

def matern32_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
    """Matérn 3/2 kernel — once differentiable, realistic roughness."""
    dist = np.sqrt(np.maximum(
        np.sum(X1**2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, axis=1),
        1e-12
    ))
    r = np.sqrt(3) * dist / length_scale
    return signal_var * (1 + r) * np.exp(-r)

def matern52_kernel(X1, X2, length_scale=1.0, signal_var=1.0):
    """Matérn 5/2 kernel — twice differentiable, a popular middle ground."""
    sq_dist = np.sum(X1**2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, axis=1)
    dist = np.sqrt(np.maximum(sq_dist, 1e-12))
    r = np.sqrt(5) * dist / length_scale
    return signal_var * (1 + r + r**2 / 3) * np.exp(-r)

def periodic_kernel(X1, X2, length_scale=1.0, signal_var=1.0, period=1.0):
    """Periodic kernel — for functions that repeat."""
    dist = np.sqrt(np.maximum(
        np.sum(X1**2, axis=1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, axis=1),
        1e-12
    ))
    return signal_var * np.exp(-2 * np.sin(np.pi * dist / period)**2 / length_scale**2)

if __name__ == "__main__":
    # Demo: compute kernel values at varying distances
    X_test = np.array([[0.0], [0.5], [1.0], [2.0], [4.0]])
    X_origin = np.array([[0.0]])

    print("Distance | RBF    | Mat3/2 | Mat5/2 | Periodic(p=2)")
    for x in X_test:
        d = float(x[0])
        r = float(rbf_kernel(x.reshape(1,-1), X_origin))
        m3 = float(matern32_kernel(x.reshape(1,-1), X_origin))
        m5 = float(matern52_kernel(x.reshape(1,-1), X_origin))
        p = float(periodic_kernel(x.reshape(1,-1), X_origin, period=2.0))
        print(f"  {d:<6.1f} | {r:.4f} | {m3:.4f} | {m5:.4f} | {p:.4f}")
