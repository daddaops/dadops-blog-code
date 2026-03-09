"""Discrete Fourier Transform from Scratch.

Implements the DFT directly (O(N^2)) and demonstrates windowing
to reduce spectral leakage. Analyzes a chord signal.
"""
import numpy as np

def dft_from_scratch(x):
    """Compute the Discrete Fourier Transform (direct O(N^2) method)."""
    N = len(x)
    n = np.arange(N)
    k = np.arange(N).reshape(-1, 1)
    # The DFT matrix: each entry is e^(-j * 2pi * k * n / N)
    W = np.exp(-2j * np.pi * k * n / N)
    return W @ x

def hann_window(N):
    """Hann window: tapers to zero at both ends."""
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))

if __name__ == "__main__":
    # Generate a chord: 330 Hz + 440 Hz + 554 Hz
    sr = 16000
    t = np.arange(512) / sr
    signal = (np.sin(2 * np.pi * 330 * t) +
              np.sin(2 * np.pi * 440 * t) +
              np.sin(2 * np.pi * 554 * t))

    # Apply window, then DFT
    windowed = signal * hann_window(512)
    X = dft_from_scratch(windowed)

    # Magnitude spectrum (only first N/2+1 bins are unique)
    magnitudes = np.abs(X[:257])
    freqs = np.arange(257) * sr / 512

    # Find peak frequencies
    threshold = np.max(magnitudes) * 0.1
    peaks = freqs[magnitudes > threshold]
    print(f"Frequency bins with significant energy: {peaks[:10].astype(int)} Hz")
    print(f"Expected peaks near: 330, 440, 554 Hz")

    # Verify against numpy FFT
    X_np = np.fft.rfft(windowed)
    max_diff = np.max(np.abs(np.abs(X[:257]) - np.abs(X_np)))
    print(f"Max difference from np.fft.rfft: {max_diff:.2e}")
