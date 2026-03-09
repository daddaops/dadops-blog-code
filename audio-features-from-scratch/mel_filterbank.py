"""Mel Filterbank from Scratch.

Builds triangular bandpass filters spaced equally on the mel scale.
Low-frequency filters are narrow (high resolution), high-frequency
filters are wide (coarse resolution), matching human hearing.
"""
import numpy as np

def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)

def mel_filterbank(n_mels=40, n_fft=512, sr=16000, f_low=0, f_high=None):
    """Build a mel-spaced triangular filterbank matrix."""
    f_high = f_high or sr / 2
    n_bins = n_fft // 2 + 1  # 257 for n_fft=512

    # M+2 equally spaced points on mel scale
    mel_points = np.linspace(hz_to_mel(f_low), hz_to_mel(f_high), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.round(hz_points * n_fft / sr).astype(int)

    filters = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        left, center, right = bin_indices[m], bin_indices[m + 1], bin_indices[m + 2]
        # Rising slope: left to center
        for k in range(left, center):
            filters[m, k] = (k - left) / (center - left)
        # Falling slope: center to right
        for k in range(center, right):
            filters[m, k] = (right - k) / (right - center)

    return filters  # Shape: (n_mels, n_bins)

if __name__ == "__main__":
    # Apply to a power spectrum
    fb = mel_filterbank(n_mels=40)
    print(f"Filterbank shape: {fb.shape}")  # (40, 257)

    power_spectrum = np.abs(np.fft.rfft(np.random.randn(512))) ** 2
    mel_energies = fb @ power_spectrum    # 40 mel-band energies
    log_mel = np.log(mel_energies + 1e-10)  # Log compression
    print(f"Mel energies shape: {mel_energies.shape}")
    print(f"Log mel range: [{log_mel.min():.2f}, {log_mel.max():.2f}]")
