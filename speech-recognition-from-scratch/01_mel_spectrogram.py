"""Simulate mel spectrogram extraction from a waveform."""
import numpy as np

if __name__ == "__main__":
    np.random.seed(42)
    sr = 16000              # 16 kHz sample rate
    duration = 3.0          # 3 seconds of speech
    n_samples = int(sr * duration)  # 48,000 raw samples

    # Generate synthetic speech-like signal (sum of formant frequencies)
    t = np.linspace(0, duration, n_samples)
    waveform = 0.5 * np.sin(2 * np.pi * 250 * t)   # F1 ~250 Hz
    waveform += 0.3 * np.sin(2 * np.pi * 2500 * t)  # F2 ~2500 Hz
    waveform += 0.1 * np.random.randn(n_samples)     # noise

    # STFT -> mel spectrogram (simplified)
    hop = sr // 100         # 10ms hop -> 100 frames/sec
    n_fft = 512
    n_frames = (n_samples - n_fft) // hop + 1
    n_mels = 80            # 80 mel frequency bins

    # Simulated mel spectrogram (in practice: STFT + mel filterbank)
    mel_spec = np.random.randn(n_frames, n_mels) * 0.5 + 2.0

    transcript = "set a timer"
    print(f"Waveform:       {n_samples:,} samples")
    print(f"Mel spectrogram: {mel_spec.shape[0]} frames x {mel_spec.shape[1]} mels")
    print(f"Transcript:      '{transcript}' ({len(transcript)} chars)")
    print(f"Mismatch:        {mel_spec.shape[0]} frames -> {len(transcript)} characters")
