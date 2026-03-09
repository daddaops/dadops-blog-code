"""Training dynamics: actual GD vs NTK-predicted loss curves.

Shows that wider networks more closely follow the NTK prediction,
and demonstrates spectral bias through eigenvalue decomposition.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def train_and_predict(X, Y, width, lr, steps, rng):
    """Train a 1-hidden-layer ReLU net, return loss curve + NTK prediction."""
    n, d = X.shape
    m = width
    W = rng.randn(m, d) * np.sqrt(2.0 / d)
    a = rng.randn(m) * np.sqrt(1.0 / m)

    # Compute initial NTK
    pre0 = X @ W.T
    h0, mask0 = relu(pre0), relu_deriv(pre0)
    J_a = h0 / np.sqrt(m)
    J_W = (a[None, :] * mask0)[:, :, None] * X[:, None, :] / np.sqrt(m)
    J = np.hstack([J_a, J_W.reshape(n, -1)])
    ntk0 = J @ J.T

    # NTK-predicted loss curve: L(t) = 0.5 * ||exp(-lr*ntk*t) * r0||^2
    f0 = relu(X @ W.T) @ a / np.sqrt(m)
    r0 = f0 - Y
    eigvals, eigvecs = np.linalg.eigh(ntk0)
    coeffs = eigvecs.T @ r0
    ntk_loss = []
    for t in range(steps):
        decay = np.exp(-lr * eigvals * t)
        ntk_loss.append(0.5 * np.sum((coeffs * decay)**2))

    # Actual gradient descent training
    losses = []
    for t in range(steps):
        pre = X @ W.T
        h = relu(pre)
        f = h @ a / np.sqrt(m)
        r = f - Y
        losses.append(0.5 * np.sum(r**2))
        # Backprop
        da = h.T @ r / np.sqrt(m)
        dW = np.zeros_like(W)
        for i in range(n):
            mask_i = relu_deriv(pre[i])
            dW += r[i] * (a * mask_i)[:, None] * X[i][None, :] / np.sqrt(m)
        a -= lr * da
        W -= lr * dW

    return losses, ntk_loss, eigvals


if __name__ == "__main__":
    rng = np.random.RandomState(7)
    X = np.linspace(-2, 2, 15).reshape(-1, 1)
    Y = np.sin(2 * X.ravel()) + 0.1 * rng.randn(15)

    for width in [20, 100, 500]:
        losses, ntk_pred, eigvals = train_and_predict(
            X, Y, width, lr=0.01 / width, steps=200, rng=np.random.RandomState(42))
        final_err = abs(losses[-1] - ntk_pred[-1]) / (ntk_pred[-1] + 1e-10)
        print(f"Width {width:>3d}: final loss={losses[-1]:.4f}, "
              f"NTK predicted={ntk_pred[-1]:.4f}, "
              f"lambda_min={eigvals[0]:.4f}")
