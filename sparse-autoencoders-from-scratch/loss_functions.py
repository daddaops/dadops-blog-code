"""SAE loss function with L1 sparsity penalty.

From: https://dadops.substack.com/sparse-autoencoders-from-scratch
"""
import numpy as np


def sae_loss(x, x_hat, z, lambda_l1):
    """
    Compute SAE loss: reconstruction + L1 sparsity.

    Returns: total_loss, recon_loss, l1_loss, l0
    """
    # Reconstruction: mean squared error
    recon_loss = np.mean((x - x_hat) ** 2)

    # Sparsity: L1 norm of hidden activations
    l1_loss = np.mean(np.sum(np.abs(z), axis=1))

    # Total loss
    total_loss = recon_loss + lambda_l1 * l1_loss

    # L0: average number of nonzero features per input
    l0 = np.mean(np.sum(z > 0, axis=1))

    return total_loss, recon_loss, l1_loss, l0


def l1_gradient(z):
    """Subgradient of L1 norm: sign(z), with 0 at z=0."""
    return np.sign(z)


if __name__ == "__main__":
    np.random.seed(42)

    # Demo: compare loss with different sparsity levels
    n, d_model, d_sae = 100, 16, 64
    x = np.random.randn(n, d_model)

    print("Effect of lambda on SAE loss components:")
    print(f"{'lambda':>8} {'total':>10} {'recon':>10} {'L1':>10} {'L0':>8}")
    print("-" * 50)

    for lam in [0.0, 0.01, 0.05, 0.1, 0.5]:
        # Simulate sparse activations
        z = np.random.randn(n, d_sae)
        z[z < 0.5] = 0  # threshold to create sparsity
        x_hat = x + np.random.randn(n, d_model) * 0.1  # noisy reconstruction

        total, recon, l1, l0 = sae_loss(x, x_hat, z, lam)
        print(f"{lam:>8.2f} {total:>10.4f} {recon:>10.4f} {l1:>10.4f} {l0:>8.1f}")

    # Demo: L1 gradient
    z_test = np.array([-2.0, -0.5, 0.0, 0.3, 1.5])
    print(f"\nL1 gradient of {z_test}: {l1_gradient(z_test)}")
