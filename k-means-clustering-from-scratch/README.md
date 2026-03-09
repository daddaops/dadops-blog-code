# K-Means Clustering from Scratch

Verified, runnable code from the [K-Means Clustering from Scratch](https://dadops.dev/blog/k-means-clustering-from-scratch/) blog post.

## Scripts

- **kmeans_basic.py** — Lloyd's K-Means with random initialization
- **kmeans_pp.py** — K-Means++ smart initialization, comparison across 20 runs
- **silhouette.py** — Silhouette score and elbow method for choosing K
- **dbscan.py** — DBSCAN density-based clustering on two-moons dataset
- **color_quantization.py** — Image color quantization using K-Means

## Run

```bash
pip install -r requirements.txt
python kmeans_basic.py
python kmeans_pp.py
python silhouette.py
python dbscan.py
python color_quantization.py
```
