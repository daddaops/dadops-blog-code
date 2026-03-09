"""Full MTL training loop with uncertainty weighting + PCGrad.

Combines all components: shared network, per-task losses, automatic
weight adjustment, and gradient conflict resolution. Uses finite-
difference gradients since blog omits backprop details.
"""
import numpy as np
from multi_task_net import MultiTaskNet
from uncertainty_weighting import UncertaintyWeighting
from pcgrad import pcgrad


def compute_grad(model, X, y_target, task='reg', eps=1e-5):
    """Finite-difference gradient of task loss w.r.t. shared W1 (flattened)."""
    grad = np.zeros_like(model.W1.ravel())
    W1_flat = model.W1.ravel()
    for idx in range(len(W1_flat)):
        old_val = W1_flat[idx]
        # Forward
        W1_flat[idx] = old_val + eps
        model.W1 = W1_flat.reshape(model.W1.shape)
        y_r, y_c, _ = model.forward(X)
        if task == 'reg':
            loss_plus = np.mean((y_r - y_target) ** 2)
        else:
            loss_plus = -np.mean(y_target * np.log(y_c + 1e-7)
                                 + (1 - y_target) * np.log(1 - y_c + 1e-7))
        # Backward
        W1_flat[idx] = old_val - eps
        model.W1 = W1_flat.reshape(model.W1.shape)
        y_r, y_c, _ = model.forward(X)
        if task == 'reg':
            loss_minus = np.mean((y_r - y_target) ** 2)
        else:
            loss_minus = -np.mean(y_target * np.log(y_c + 1e-7)
                                  + (1 - y_target) * np.log(1 - y_c + 1e-7))
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        W1_flat[idx] = old_val
    model.W1 = W1_flat.reshape(model.W1.shape)
    return grad


def apply_gradients(model, update_dir, lr):
    """Apply gradient update to shared W1 parameters."""
    model.W1 -= lr * update_dir.reshape(model.W1.shape)


def train_mtl(model, X, y_reg, y_cls, steps=1000, lr=0.001):
    """Full MTL training loop with uncertainty weighting + PCGrad."""
    uw = UncertaintyWeighting(n_tasks=2)
    history = {'reg': [], 'cls': [], 'w_reg': [], 'w_cls': []}

    for step in range(steps):
        y_r, y_c, h2 = model.forward(X)
        # Compute per-task losses
        loss_reg = np.mean((y_r - y_reg) ** 2)
        eps = 1e-7
        loss_cls = -np.mean(y_cls * np.log(y_c + eps)
                            + (1 - y_cls) * np.log(1 - y_c + eps))
        # Update uncertainty weights
        uw.update([loss_reg, loss_cls], lr=0.01)
        weights = uw.effective_weights()
        # Compute per-task gradients (via backprop, shown conceptually)
        g_reg = compute_grad(model, X, y_reg, task='reg')
        g_cls = compute_grad(model, X, y_cls, task='cls')
        # Apply PCGrad to resolve conflicts
        update_dir = pcgrad([weights[0] * g_reg, weights[1] * g_cls])
        # Update shared parameters
        apply_gradients(model, update_dir, lr=lr)
        # Log
        history['reg'].append(loss_reg)
        history['cls'].append(loss_cls)
        history['w_reg'].append(weights[0])
        history['w_cls'].append(weights[1])
        if step % 200 == 0:
            cos_sim = np.dot(g_reg, g_cls) / (
                np.linalg.norm(g_reg) * np.linalg.norm(g_cls) + eps)
            print(f"Step {step}: L_reg={loss_reg:.3f}, L_cls={loss_cls:.3f}, "
                  f"w=[{weights[0]:.3f}, {weights[1]:.3f}], cos={cos_sim:.3f}")
    return history


if __name__ == "__main__":
    np.random.seed(42)
    # Small dataset for finite-difference feasibility
    n, d_in, d_hidden = 20, 4, 8
    X = np.random.randn(n, d_in)
    y_reg = 3.0 * X[:, 0] - 2.0 * X[:, 1] + np.random.randn(n) * 0.3
    y_cls = (X[:, 2] + X[:, 3] > 0).astype(float)

    model = MultiTaskNet(d_in=d_in, d_hidden=d_hidden)
    history = train_mtl(model, X, y_reg, y_cls, steps=5, lr=0.001)

    print(f"\nTraining complete: {len(history['reg'])} steps")
    print(f"Final L_reg={history['reg'][-1]:.3f}")
    print(f"Final L_cls={history['cls'][-1]:.3f}")
    print(f"Final weights=[{history['w_reg'][-1]:.4f}, {history['w_cls'][-1]:.4f}]")
