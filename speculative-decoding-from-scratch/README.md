# Speculative Decoding from Scratch — Code Extracts

Standalone Python scripts extracted from the [DadOps blog post](https://dadops.co/blog/speculative-decoding-from-scratch/).

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | `01_arithmetic_intensity.py` | Arithmetic intensity analysis comparing decode vs prefill on an A100 GPU |
| 2 | `02_speculative_decode_step.py` | Single speculative decoding step with toy draft/target models |
| 3 | `03_rejection_sampling.py` | Rejection sampling mechanism with empirical validation |
| 4 | `04_rejection_sampling_walkthrough.py` | Step-by-step math walkthrough proving losslessness |
| 5 | `05_full_decode_loop.py` | Complete speculative decoding loop with toy bigram models |
| 6 | `06_expected_speedup.py` | Expected speedup analysis across acceptance rates and K values |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python 01_arithmetic_intensity.py
python 02_speculative_decode_step.py
python 03_rejection_sampling.py
python 04_rejection_sampling_walkthrough.py
python 05_full_decode_loop.py
python 06_expected_speedup.py
```
