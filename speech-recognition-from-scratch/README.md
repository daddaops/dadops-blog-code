# Speech Recognition from Scratch — Code Extracts

Python code blocks extracted from the DadOps blog post
[Speech Recognition from Scratch](https://dadops.dev/blog/speech-recognition-from-scratch/).

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | `01_mel_spectrogram.py` | Simulate mel spectrogram extraction from a waveform |
| 2 | `02_ctc_forward.py` | CTC forward algorithm to compute CTC loss |
| 3 | `03_ctc_decoding.py` | CTC decoding: greedy vs beam search with prefix merging |
| 4 | `04_attention_mechanism.py` | Dot-product attention for encoder-decoder ASR |
| 5 | `05_word_error_rate.py` | Word Error Rate (WER) via dynamic programming |
| 6 | `06_asr_pipeline.py` | End-to-end ASR pipeline simulation |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python 01_mel_spectrogram.py
python 02_ctc_forward.py
python 03_ctc_decoding.py
python 04_attention_mechanism.py
python 05_word_error_rate.py
python 06_asr_pipeline.py
```
