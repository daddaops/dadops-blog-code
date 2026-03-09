# Conformal Prediction from Scratch

Code extracted from the blog post: [Conformal Prediction from Scratch](https://dadops.co/blog/conformal-prediction-from-scratch/)

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | `01_nonconformity_scores.py` | Computing calibration scores from softmax outputs |
| 2 | `02_split_conformal.py` | Split conformal prediction — the core algorithm |
| 3 | `03_adaptive_prediction_sets.py` | Adaptive Prediction Sets (APS) — difficulty-aware sets |
| 4 | `04_conformalized_quantile_regression.py` | CQR — adaptive-width prediction intervals for regression |
| 5 | `05_mondrian_conformal.py` | Mondrian conformal — per-group coverage for imbalanced classes |
| 6 | `06_coverage_efficiency_tradeoff.py` | Sweeping alpha to see coverage vs. efficiency tradeoff |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python 01_nonconformity_scores.py
python 02_split_conformal.py
python 03_adaptive_prediction_sets.py
python 04_conformalized_quantile_regression.py
python 05_mondrian_conformal.py
python 06_coverage_efficiency_tradeoff.py
```
