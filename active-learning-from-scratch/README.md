# Active Learning from Scratch

Verified, runnable code from the [Active Learning from Scratch](https://dadops.dev/blog/active-learning-from-scratch/) blog post.

## Scripts

| Script | Description |
|--------|-------------|
| `uncertainty_sampling.py` | Three acquisition functions (least confidence, margin, entropy) with AL loop |
| `query_by_committee.py` | QBC with bootstrapped logistic regression committee |
| `expected_gradient_length.py` | EGL scoring with a simple PyTorch neural network |
| `batch_selection.py` | Top-K vs K-Centers vs Hybrid batch selection strategies |
| `hybrid_active_learning.py` | Epsilon-greedy AL to avoid sampling bias |

## Quick Start

```bash
pip install -r requirements.txt
python uncertainty_sampling.py
python query_by_committee.py
python expected_gradient_length.py
python batch_selection.py
python hybrid_active_learning.py
```

## Dependencies

- numpy
- scikit-learn
- scipy
- torch (CPU only, for EGL)
