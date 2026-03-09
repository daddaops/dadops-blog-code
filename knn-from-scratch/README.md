# K-Nearest Neighbors from Scratch

Verified, runnable code from the [K-Nearest Neighbors from Scratch](https://dadops.dev/blog/knn-from-scratch/) blog post.

## Scripts

- **one_nn.py** — 1-Nearest Neighbor classifier on spiral data (shared module)
- **knn_classifier.py** — KNN classifier with k sweep (k=1,5,15,50)
- **distance_metrics.py** — Euclidean, Manhattan, Minkowski, Chebyshev, cosine distances + feature scaling
- **curse_of_dimensionality.py** — Nearest/farthest ratio collapse in high dimensions
- **knn_regression.py** — KNN regressor with uniform and distance weighting
- **kdtree.py** — KD-tree construction and benchmark vs brute force

## Run

```bash
pip install -r requirements.txt
python one_nn.py
python knn_classifier.py
python distance_metrics.py
python curse_of_dimensionality.py
python knn_regression.py
python kdtree.py
```
