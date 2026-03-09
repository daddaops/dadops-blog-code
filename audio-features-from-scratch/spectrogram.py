"""Short-Time Fourier Transform (STFT) and Spectrogram from Scratch.

Slides a window across the signal and computes FFT on each frame,
producing a 2D time-frequency representation.
"""
import numpy as np

def stft(signal, sr=16000, win_ms=25, hop_ms=10, n_fft=512):
    """Short-Time Fourier Transform from scratch."""
    win_len = int(sr * win_ms / 1000)  # 400 samples at 16 kHz
    hop_len = int(sr * hop_ms / 1000)  # 160 samples at 16 kHz

    # Hann window
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_len) / (win_len - 1))

    # Number of frames
    n_frames = 1 + (len(signal) - win_len) // hop_len
    spectrogram = np.zeros((n_fft // 2 + 1, n_frames))

    for i in range(n_frames):
        start = i * hop_len
        frame = signal[start:start + win_len] * window
        # Zero-pad to n_fft and compute FFT
        padded = np.zeros(n_fft)
        padded[:win_len] = frame
        spectrum = np.fft.rfft(padded)
        spectrogram[:, i] = np.abs(spectrum) ** 2  # Power spectrum

    return spectrogram

if __name__ == "__main__":
    # Compute log-power spectrogram
    signal = np.random.randn(16000)  # 1 second of noise
    S = stft(signal)
    log_S = 10 * np.log10(S + 1e-10)  # Log-power in dB
    print(f"Spectrogram shape: {S.shape}")  # (257 frequency bins, 100 time frames)
    print(f"Log-power range: [{log_S.min():.1f}, {log_S.max():.1f}] dB")
