# CRF from Scratch - Code Blocks

Extracted from the [Conditional Random Fields from Scratch](../../blog/crf-from-scratch/) blog post.

## Scripts

| # | File | Description |
|---|------|-------------|
| 1 | `01_crf_scoring.py` | LinearChainCRF class with emission and sequence scoring |
| 2 | `02_forward_algorithm.py` | Forward algorithm for log-partition function, verified against brute force |
| 3 | `03_viterbi_decode.py` | Viterbi decoding to find the highest-scoring label sequence |
| 4 | `04_forward_backward.py` | Forward-backward algorithm for node/edge marginals + train_crf function |
| 5 | `05_train_crf.py` | End-to-end training on synthetic POS tagging data with evaluation |

## Requirements

```
pip install -r requirements.txt
```

Dependencies: numpy, scipy

## Running

Each script is self-contained and runnable independently:

```bash
python3 01_crf_scoring.py
python3 02_forward_algorithm.py
python3 03_viterbi_decode.py
python3 04_forward_backward.py
python3 05_train_crf.py
```
