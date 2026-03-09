"""Masked Autoencoder (MAE) for images."""
import numpy as np

# Simulate a 4x4 "image" as 16 patches (each patch = 8-dim feature vector)
np.random.seed(42)
patches = np.random.randn(16, 8)

# Add spatial structure: top patches = "sky", bottom patches = "ground"
for i in range(8):       # rows 0-1 (top half)
    patches[i] += 2.0
for i in range(8, 16):   # rows 2-3 (bottom half)
    patches[i] -= 2.0

def masked_autoencoder(patches, mask_ratio=0.75):
    n = len(patches)
    n_mask = int(n * mask_ratio)

    # Random shuffle to split into masked and visible
    perm = np.random.permutation(n)
    mask_idx = perm[:n_mask]
    vis_idx = perm[n_mask:]

    # KEY INSIGHT 1: Encoder only processes VISIBLE patches (saves compute!)
    visible = patches[vis_idx]
    W_enc = np.random.randn(8, 16) * 0.1
    encoded = np.tanh(visible @ W_enc)  # (n_visible, 16)

    # Decoder: reconstruct ALL patches from encoded visible ones
    mask_token = np.zeros(16)  # learnable placeholder for masked positions
    full_seq = np.zeros((n, 16))
    full_seq[vis_idx] = encoded
    full_seq[mask_idx] = mask_token

    W_dec = np.random.randn(16, 8) * 0.1
    reconstructed = full_seq @ W_dec  # (n, 8)

    # KEY INSIGHT 2: Loss only on MASKED patches
    mse = np.mean((reconstructed[mask_idx] - patches[mask_idx]) ** 2)
    return mse, len(vis_idx), len(mask_idx)

# Compare reconstruction difficulty at different masking ratios
for ratio in [0.25, 0.50, 0.75, 0.90]:
    results = [masked_autoencoder(patches, ratio) for _ in range(10)]
    avg_mse = np.mean([r[0] for r in results])
    n_vis, n_mask = results[0][1], results[0][2]
    print(f"Mask {ratio:.0%}: see {n_vis} patches, reconstruct {n_mask} -> MSE={avg_mse:.2f}")
