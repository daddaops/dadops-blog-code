"""Demonstrates the problem with uniform task weighting in MTL.

When regression loss (~100s) and classification loss (~0.7) are weighted
equally, the larger regression gradients dominate shared parameters.
"""
import numpy as np
from multi_task_net import MultiTaskNet


def train_uniform(model, X, y_reg, y_cls, lr=0.001, steps=500):
    """Train with equal weighting — regression dominates."""
    losses_reg, losses_cls = [], []
    for step in range(steps):
        y_r, y_c, h2 = model.forward(X)
        # Regression: MSE (loss scale ~100s)
        loss_reg = np.mean((y_r - y_reg) ** 2)
        # Classification: binary cross-entropy (loss scale ~0.7)
        eps = 1e-7
        loss_cls = -np.mean(y_cls * np.log(y_c + eps)
                            + (1 - y_cls) * np.log(1 - y_c + eps))
        # Uniform weighting: L = L_reg + L_cls
        # Gradient of L_reg overwhelms gradient of L_cls
        losses_reg.append(loss_reg)
        losses_cls.append(loss_cls)
        # ... backprop and update (omitted for brevity)
    return losses_reg, losses_cls
# Typical result: regression loss drops fast, classification barely moves


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y_reg = 10.0 * X[:, 0] + 5.0 * X[:, 1] + np.random.randn(100) * 0.5
    y_cls = (X[:, 2] + X[:, 3] > 0).astype(float)

    model = MultiTaskNet(d_in=8, d_hidden=32)
    losses_reg, losses_cls = train_uniform(model, X, y_reg, y_cls)

    print("Uniform weighting (no backprop — forward loss tracking only):")
    print(f"  Step 0:   L_reg={losses_reg[0]:.3f}, L_cls={losses_cls[0]:.3f}")
    print(f"  Step 499: L_reg={losses_reg[-1]:.3f}, L_cls={losses_cls[-1]:.3f}")
    print(f"  L_reg scale: ~{losses_reg[0]:.0f} vs L_cls scale: ~{losses_cls[0]:.1f}")
    print(f"  Ratio: {losses_reg[0] / losses_cls[0]:.0f}x — regression dominates!")
