# ML Evaluation from Scratch

Verified, runnable code from the [DadOps blog post](https://dadops.co/blog/ml-evaluation-from-scratch/).

## Scripts

| Script | Description |
|--------|-------------|
| `evaluation_splits.py` | Random, stratified, and temporal splitting + variance demo |
| `cross_validation.py` | K-fold and stratified k-fold cross-validation |
| `metrics.py` | Confusion matrix, precision/recall/F1, MCC, ROC-AUC |
| `significance.py` | Paired t-test, corrected Nadeau-Bengio test, McNemar's test |

## Usage

```bash
python evaluation_splits.py   # Variance of 20 random splits
python cross_validation.py    # 10-fold CV tighter variance
python metrics.py             # Metrics on imbalanced fraud data
python significance.py        # Naive vs corrected significance tests
```

Dependencies: numpy, scikit-learn, scipy.
