"""Synthetic Waveform Generation and Aliasing Demonstration.

Generates pure sine, chord, and chirp signals. Demonstrates aliasing
when sampling above Nyquist frequency.
"""
import numpy as np

def generate_waveforms(sr=16000, duration=0.05):
    """Generate three synthetic audio signals."""
    t = np.arange(int(sr * duration)) / sr

    # Pure 440 Hz sine wave (concert A)
    sine = np.sin(2 * np.pi * 440 * t)

    # Three-tone chord: 330 Hz (E4) + 440 Hz (A4) + 554 Hz (C#5)
    chord = (np.sin(2 * np.pi * 330 * t) +
             np.sin(2 * np.pi * 440 * t) +
             np.sin(2 * np.pi * 554 * t)) / 3

    # Linear chirp: sweeps 200 Hz to 2000 Hz
    freq = 200 + (2000 - 200) * t / t[-1]
    chirp = np.sin(2 * np.pi * np.cumsum(freq) / sr)

    return t, sine, chord, chirp

if __name__ == "__main__":
    t, sine, chord, chirp = generate_waveforms()
    print(f"Generated {len(t)} samples at 16kHz ({t[-1]*1000:.1f} ms)")
    print(f"Sine range: [{sine.min():.3f}, {sine.max():.3f}]")
    print(f"Chord range: [{chord.min():.3f}, {chord.max():.3f}]")
    print(f"Chirp range: [{chirp.min():.3f}, {chirp.max():.3f}]")

    # Aliasing demonstration: sample a 7 kHz tone at 10 kHz
    # Nyquist frequency = 5 kHz, so 7 kHz aliases to 10 - 7 = 3 kHz
    sr_low = 10000
    t_low = np.arange(500) / sr_low
    original_7k = np.sin(2 * np.pi * 7000 * t_low)
    alias_3k = np.sin(2 * np.pi * 3000 * t_low)
    print(f"\nAliasing demo: max diff between 7kHz@10kHz and 3kHz: {np.max(np.abs(original_7k - alias_3k)):.6f}")
