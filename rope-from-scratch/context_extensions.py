"""RoPE context extension techniques: PI, NTK, YaRN."""
import numpy as np
from helpers import rope_frequencies


def rope_frequencies_pi(d_model, base=10000.0, scale=1.0):
    """Position Interpolation: scale down positions to fit training range."""
    i = np.arange(0, d_model, 2, dtype=np.float64)
    freqs = 1.0 / (base ** (i / d_model))
    return freqs * scale


def rope_frequencies_ntk(d_model, base=10000.0, scale_factor=2.0):
    """NTK-aware scaling: modify the base to extend context."""
    base_scaled = base * (scale_factor ** (d_model / (d_model - 2)))
    i = np.arange(0, d_model, 2, dtype=np.float64)
    return 1.0 / (base_scaled ** (i / d_model))


def rope_frequencies_yarn(d_model, base=10000.0, scale_factor=2.0,
                           original_max_len=4096, alpha=1, beta=32):
    """YaRN: piecewise frequency scaling with temperature correction."""
    i = np.arange(0, d_model, 2, dtype=np.float64)
    freqs = 1.0 / (base ** (i / d_model))

    wavelengths = 2 * np.pi / freqs
    ratios = original_max_len / wavelengths

    gamma = np.clip((ratios - alpha) / (beta - alpha), 0.0, 1.0)

    freqs_scaled = freqs / scale_factor
    freqs_yarn = gamma * freqs + (1 - gamma) * freqs_scaled

    temperature = 0.1 * np.log(scale_factor) + 1.0

    return freqs_yarn, temperature


# Compare all methods
d_model = 64
freqs_base = rope_frequencies(d_model)
freqs_pi = rope_frequencies_pi(d_model, scale=0.5)
freqs_ntk = rope_frequencies_ntk(d_model, scale_factor=2.0)
freqs_yarn, temp = rope_frequencies_yarn(d_model, scale_factor=2.0)

print(f"Base freqs  [0]: {freqs_base[0]:.4f}, [-1]: {freqs_base[-1]:.6f}")
print(f"PI freqs    [0]: {freqs_pi[0]:.4f}, [-1]: {freqs_pi[-1]:.6f}")
print(f"NTK freqs   [0]: {freqs_ntk[0]:.4f}, [-1]: {freqs_ntk[-1]:.6f}")
print(f"YaRN freqs  [0]: {freqs_yarn[0]:.4f}, [-1]: {freqs_yarn[-1]:.6f}")
print(f"YaRN temperature: {temp:.4f}")
