# Anomaly Detection from Scratch

Verified, runnable code from the DadOps blog post:
[Anomaly Detection from Scratch](https://daddaops.com/blog/anomaly-detection-from-scratch/)

## Scripts

| Script | Description |
|--------|-------------|
| `statistical_detection.py` | Z-Score, MAD, and Grubbs' Test on server response times |
| `knn_detector.py` | k-NN distance-based anomaly scoring on 2D clustered data |
| `lof.py` | Local Outlier Factor — density-adaptive anomaly detection |
| `isolation_forest.py` | Isolation Forest — random tree-based anomaly detection in 10D |
| `benchmark.py` | Z-score benchmark across three scenarios |

## Usage

```bash
pip install -r requirements.txt  # no external deps needed
python3 statistical_detection.py
python3 knn_detector.py
python3 lof.py
python3 isolation_forest.py
python3 benchmark.py
```

All scripts use seeded RNGs for reproducible output.
