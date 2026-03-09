# Audio Features from Scratch

Verified, runnable code from the DadOps blog post:
[Audio Features from Scratch](https://daddaops.com/blog/audio-features-from-scratch/)

## Scripts

| Script | Description |
|--------|-------------|
| `waveforms.py` | Synthetic waveform generation + aliasing demo |
| `dft.py` | Discrete Fourier Transform (O(N^2)) + windowing |
| `spectrogram.py` | STFT and log-power spectrogram |
| `mel_filterbank.py` | Mel-spaced triangular filterbank matrix |
| `mfcc.py` | Full MFCC extraction pipeline |
| `spectral_features.py` | Spectral centroid, bandwidth, and SpecAugment |

## Usage

```bash
pip install -r requirements.txt
python3 waveforms.py
python3 dft.py
python3 spectrogram.py
python3 mel_filterbank.py
python3 mfcc.py
python3 spectral_features.py
```
