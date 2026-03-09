"""Adjoint method for computing gradients through a Neural ODE.

O(1) memory: stores only the trajectory, then integrates the
adjoint ODE backward to accumulate parameter gradients.
"""
import numpy as np
from neural_ode import NeuralODE


def adjoint_gradients(node, h0, target, t0=0.0, t1=1.0, steps=20):
    """Compute dL/d(params) via the adjoint method (O(1) memory)."""
    # Forward solve: store trajectory for reconstruction
    h_final, trajectory = node.forward(h0, t0, t1, steps)

    # Loss: L = 0.5 * ||h(T) - target||^2
    loss = 0.5 * np.sum((h_final - target) ** 2)

    # Terminal condition: a(T) = dL/dh(T)
    a = h_final - target

    # Backward solve: integrate adjoint ODE from T to 0
    dt = (t1 - t0) / steps
    grad_W1 = np.zeros_like(node.W1)
    grad_W2 = np.zeros_like(node.W2)

    for i in range(steps - 1, -1, -1):
        h = trajectory[i]
        z = np.tanh(node.W1 @ h + node.b1)
        dz = 1 - z ** 2  # tanh derivative

        # Jacobian: df/dh = W2 @ diag(dz) @ W1
        df_dh = node.W2 @ np.diag(dz) @ node.W1

        # Accumulate parameter gradients
        grad_W2 += np.outer(a, z) * dt
        grad_W1 += np.outer(node.W2.T @ a * dz, h) * dt

        # Adjoint update (Euler step backward)
        a = a + dt * (df_dh.T @ a)

    return loss, {'W1': grad_W1, 'W2': grad_W2}


if __name__ == "__main__":
    np.random.seed(42)
    node = NeuralODE(dim=2, hidden=8)
    h0 = np.array([1.0, 0.5])
    target = np.array([0.0, 1.0])

    loss, grads = adjoint_gradients(node, h0, target)
    print(f"Loss: {loss:.4f}")
    print(f"grad_W1 shape: {grads['W1'].shape}, norm: {np.linalg.norm(grads['W1']):.4f}")
    print(f"grad_W2 shape: {grads['W2'].shape}, norm: {np.linalg.norm(grads['W2']):.4f}")

    # Verify gradient with finite differences
    eps = 1e-5
    fd_grad = np.zeros_like(node.W2)
    for i in range(node.W2.shape[0]):
        for j in range(node.W2.shape[1]):
            node.W2[i, j] += eps
            h_p, _ = node.forward(h0)
            loss_p = 0.5 * np.sum((h_p - target) ** 2)
            node.W2[i, j] -= 2 * eps
            h_m, _ = node.forward(h0)
            loss_m = 0.5 * np.sum((h_m - target) ** 2)
            node.W2[i, j] += eps
            fd_grad[i, j] = (loss_p - loss_m) / (2 * eps)

    print(f"\nGradient check (W2):")
    print(f"  Adjoint norm:  {np.linalg.norm(grads['W2']):.6f}")
    print(f"  Finite diff:   {np.linalg.norm(fd_grad):.6f}")
    print(f"  Relative error: {np.linalg.norm(grads['W2'] - fd_grad) / (np.linalg.norm(fd_grad) + 1e-8):.6f}")
