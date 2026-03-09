"""NormalFloat4 (NF4) quantization vs uniform INT4."""
import numpy as np
from scipy.stats import norm

def compute_nf4_levels():
    """Compute NormalFloat4 quantization levels.

    Place 16 levels at quantiles of N(0,1) so each bin has
    equal probability mass. Then normalize to [-1, 1] and
    ensure there's an exact zero."""
    n_levels = 16
    # 8 negative levels, 1 zero, 7 positive = 16 total
    # Negative side: 8 quantiles of the negative half
    neg_levels = [norm.ppf((i + 0.5) / (2 * 8)) for i in range(8)]
    # Positive side: 7 quantiles of the positive half (zero handled separately)
    pos_levels = [norm.ppf(0.5 + (i + 0.5) / (2 * 8)) for i in range(1, 8)]

    levels = neg_levels + [0.0] + pos_levels
    levels = sorted(levels)

    # Normalize to [-1, 1]
    max_abs = max(abs(l) for l in levels)
    levels = [l / max_abs for l in levels]

    return np.array(levels)

def compute_int4_levels():
    """Standard INT4 levels, normalized to [-1, 1]."""
    levels = np.linspace(-1, 1, 16)
    return levels

nf4 = compute_nf4_levels()
int4 = compute_int4_levels()

print("NF4 levels (normalized):")
print("  ", [f"{l:+.4f}" for l in nf4])
print("\nINT4 levels (uniform):")
print("  ", [f"{l:+.4f}" for l in int4])

# Compare quantization error on normally-distributed weights
np.random.seed(42)
weights = np.random.randn(10000).astype(np.float32)

def quantize_with_levels(w, levels):
    """Map each weight to the nearest level."""
    scale = np.max(np.abs(w))
    normalized = w / scale
    # Find nearest level for each weight
    indices = np.argmin(np.abs(normalized[:, None] - levels[None, :]), axis=1)
    dequantized = levels[indices] * scale
    return dequantized

deq_nf4 = quantize_with_levels(weights, nf4)
deq_int4 = quantize_with_levels(weights, int4)

mse_nf4 = np.mean((weights - deq_nf4) ** 2)
mse_int4 = np.mean((weights - deq_int4) ** 2)

print(f"\nMSE on N(0,1) weights:")
print(f"  INT4 (uniform):      {mse_int4:.6f}")
print(f"  NF4  (normal-aware): {mse_nf4:.6f}")
print(f"  NF4 improvement:     {(1 - mse_nf4/mse_int4)*100:.1f}%")
