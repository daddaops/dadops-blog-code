"""MFCC Extraction Pipeline from Scratch.

Complete pipeline: pre-emphasis -> frame -> window -> FFT -> power spectrum
-> mel filterbank -> log -> DCT -> 13 cepstral coefficients per frame.
"""
import numpy as np
from mel_filterbank import mel_filterbank

def preemphasis(signal, coeff=0.97):
    """High-pass filter to flatten speech spectrum."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def dct_matrix(n_mfcc, n_mels):
    """Type-II DCT matrix for cepstral transform."""
    n = np.arange(n_mels)
    k = np.arange(n_mfcc).reshape(-1, 1)
    return np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))

def extract_mfccs(signal, sr=16000, n_mfcc=13, n_mels=40):
    """Full MFCC pipeline from scratch."""
    # Step 1: Pre-emphasis
    emphasized = preemphasis(signal)

    # Step 2: Frame and window (25 ms window, 10 ms hop)
    win_len, hop_len, n_fft = 400, 160, 512
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(win_len) / (win_len - 1))
    n_frames = 1 + (len(emphasized) - win_len) // hop_len

    mfccs = np.zeros((n_mfcc, n_frames))
    fb = mel_filterbank(n_mels, n_fft, sr)
    dct = dct_matrix(n_mfcc, n_mels)

    for i in range(n_frames):
        frame = emphasized[i * hop_len : i * hop_len + win_len] * window
        # Step 3: FFT and power spectrum
        padded = np.zeros(n_fft)
        padded[:win_len] = frame
        power = np.abs(np.fft.rfft(padded)) ** 2
        # Step 4: Mel filterbank, log, DCT
        mel_energy = fb @ power
        log_mel = np.log(mel_energy + 1e-10)
        mfccs[:, i] = dct @ log_mel

    return mfccs  # Shape: (13, n_frames)

if __name__ == "__main__":
    # Extract MFCCs from 1 second of synthetic speech
    signal = np.random.randn(16000)
    mfccs = extract_mfccs(signal)
    print(f"MFCC shape: {mfccs.shape}")  # (13, 100)
