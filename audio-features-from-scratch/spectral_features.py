"""Spectral Features and SpecAugment from Scratch.

Implements spectral centroid, bandwidth, and SpecAugment data augmentation.
"""
import numpy as np

def spectral_centroid(power_spectrum, freqs):
    """Center of mass of the power spectrum."""
    return np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)

def spectral_bandwidth(power_spectrum, freqs, centroid):
    """Spread of the spectrum around its centroid."""
    deviation = (freqs - centroid) ** 2
    return np.sqrt(np.sum(deviation * power_spectrum) / (np.sum(power_spectrum) + 1e-10))

def spec_augment(mel_spec, freq_mask_param=15, time_mask_param=25,
                 n_freq_masks=2, n_time_masks=2):
    """SpecAugment: frequency and time masking on mel spectrogram."""
    augmented = mel_spec.copy()
    n_mels, n_frames = augmented.shape

    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        augmented[f0:f0 + f, :] = 0  # Zero out frequency bands

    for _ in range(n_time_masks):
        t = np.random.randint(0, time_mask_param + 1)
        t0 = np.random.randint(0, max(1, n_frames - t))
        augmented[:, t0:t0 + t] = 0  # Zero out time steps

    return augmented

if __name__ == "__main__":
    # Spectral features on a random spectrum
    n_fft = 512
    sr = 16000
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    power_spectrum = np.abs(np.fft.rfft(np.random.randn(n_fft))) ** 2

    centroid = spectral_centroid(power_spectrum, freqs)
    bandwidth = spectral_bandwidth(power_spectrum, freqs, centroid)
    print(f"Spectral centroid: {centroid:.0f} Hz")
    print(f"Spectral bandwidth: {bandwidth:.0f} Hz")

    # SpecAugment example
    mel = np.random.rand(40, 100)  # 40 mel bands, 100 frames
    augmented = spec_augment(mel)
    n_zeros = np.sum(augmented == 0)
    print(f"\nSpecAugment: {n_zeros} values masked out of {mel.size}")
