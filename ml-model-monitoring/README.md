# ML Model Monitoring — Code from Blog Post

Extracted from: https://dadops.dev/blog/ml-model-monitoring/

## Scripts

| Script | Code Block | Description |
|--------|-----------|-------------|
| `drift_detection.py` | 1 | Data drift metrics — PSI, KS statistic, JS divergence |
| `prediction_quality.py` | 2 | Prediction quality — ECE (calibration error), prediction distribution shift |
| `feature_health.py` | 3 | Feature health monitor — null rates, mean shifts, range violations, new categories |
| `cusum_detector.py` | 4 | CUSUM vs fixed-threshold alerting — detects gradual drift earlier |
| `model_monitor.py` | 5 | Complete monitoring pipeline — ties drift, quality, features, and alerting together |

## Dependencies

- Python 3.8+
- All scripts are stdlib-only (random, math) — no external dependencies

## Running

```bash
# Run individual scripts
python drift_detection.py
python prediction_quality.py
python feature_health.py
python cusum_detector.py
python model_monitor.py
```
