"""L-BFGS two-loop recursion on an ill-conditioned quadratic."""
import numpy as np

def lbfgs_two_loop(grad, s_history, y_history):
    """Compute H_k * grad using L-BFGS two-loop recursion."""
    m = len(s_history)
    q = grad.copy()
    alphas = np.zeros(m)
    rhos = np.zeros(m)

    # Backward loop: peel off each correction
    for i in range(m - 1, -1, -1):
        rhos[i] = 1.0 / (y_history[i] @ s_history[i])
        alphas[i] = rhos[i] * (s_history[i] @ q)
        q = q - alphas[i] * y_history[i]

    # Initial Hessian estimate: scale by most recent curvature
    if m > 0:
        gamma = (s_history[-1] @ y_history[-1]) / (y_history[-1] @ y_history[-1])
        r = gamma * q
    else:
        r = q

    # Forward loop: apply corrections in order
    for i in range(m):
        beta = rhos[i] * (y_history[i] @ r)
        r = r + (alphas[i] - beta) * s_history[i]

    return r

# Demo: L-BFGS on a 10D ill-conditioned quadratic
n = 10
np.random.seed(42)
eigvals = np.logspace(0, 3, n)  # condition number = 1000
Q, _ = np.linalg.qr(np.random.randn(n, n))
A = Q @ np.diag(eigvals) @ Q.T

x = np.ones(n)
s_hist, y_hist = [], []
m_keep = 10  # number of pairs to store

for step in range(50):
    g = A @ x
    if np.linalg.norm(g) < 1e-10:
        print(f"L-BFGS converged in {step} steps (kappa=1000, n={n})")
        break
    direction = lbfgs_two_loop(g, s_hist, y_hist)
    x_new = x - direction  # step size = 1 for quadratics
    s_hist.append(x_new - x)
    y_hist.append(A @ x_new - g)
    if len(s_hist) > m_keep:
        s_hist.pop(0)
        y_hist.pop(0)
    x = x_new
# Output: L-BFGS converged in 11 steps (kappa=1000, n=10)
