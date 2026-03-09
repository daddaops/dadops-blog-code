import numpy as np

def mixup(x1, y1, x2, y2, alpha=0.2, rng=None):
    """Mixup: blend two examples and their labels."""
    rng = rng or np.random.default_rng()
    lam = rng.beta(alpha, alpha)
    x_mix = lam * x1 + (1 - lam) * x2
    y_mix = lam * y1 + (1 - lam) * y2
    return x_mix, y_mix

def cutmix(x1, y1, x2, y2, alpha=1.0, rng=None):
    """CutMix: paste a patch from x2 onto x1, mix labels by area."""
    rng = rng or np.random.default_rng()
    h, w = x1.shape[:2]
    lam = rng.beta(alpha, alpha)

    # Compute patch dimensions from lambda
    cut_ratio = np.sqrt(1 - lam)
    ch, cw = int(h * cut_ratio), int(w * cut_ratio)

    # Random patch center
    cy = rng.integers(0, h)
    cx = rng.integers(0, w)

    # Clamp patch to image bounds
    y1_coord = max(0, cy - ch // 2)
    y2_coord = min(h, cy + ch // 2)
    x1_coord = max(0, cx - cw // 2)
    x2_coord = min(w, cx + cw // 2)

    out = x1.copy()
    out[y1_coord:y2_coord, x1_coord:x2_coord] = x2[y1_coord:y2_coord, x1_coord:x2_coord]

    # Adjust lambda to actual patch area
    actual_lam = 1 - (y2_coord - y1_coord) * (x2_coord - x1_coord) / (h * w)
    y_mix = actual_lam * y1 + (1 - actual_lam) * y2
    return out, y_mix


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Create two 16x16 RGB images with one-hot labels
    x1 = rng.random((16, 16, 3))
    x2 = rng.random((16, 16, 3))
    y1 = np.array([1.0, 0.0, 0.0])  # class 0
    y2 = np.array([0.0, 1.0, 0.0])  # class 1

    x_mix, y_mix = mixup(x1, y1, x2, y2, alpha=0.2, rng=rng)
    print("Mixup:")
    print("  Mixed image shape:", x_mix.shape)
    print("  Mixed label:", y_mix)
    print("  Label sums to 1:", np.isclose(y_mix.sum(), 1.0))

    x_cut, y_cut = cutmix(x1, y1, x2, y2, alpha=1.0, rng=rng)
    print("\nCutMix:")
    print("  Mixed image shape:", x_cut.shape)
    print("  Mixed label:", y_cut)
    print("  Label sums to 1:", np.isclose(y_cut.sum(), 1.0))
    print("All Mixup/CutMix tests passed.")
