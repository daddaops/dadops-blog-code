# Hierarchical Clustering from Scratch

Verified, runnable code from the [Hierarchical Clustering from Scratch](https://dadops.dev/blog/hierarchical-clustering-from-scratch/) blog post.

## Scripts

- **agglomerative_basic.py** — Basic agglomerative clustering with single linkage
- **lance_williams.py** — Configurable linkage via Lance-Williams formula (single, complete, average, ward)
- **dendrogram.py** — Dendrogram leaf ordering and cophenetic correlation
- **dendrogram_cutting.py** — Flat cluster extraction and inconsistency-based cutting
- **mst_single_linkage.py** — MST-based single linkage via Kruskal's algorithm
- **benchmark.py** — Performance comparison with scipy (requires scipy)

## Run

```bash
pip install -r requirements.txt
python agglomerative_basic.py
python lance_williams.py
python dendrogram.py
python dendrogram_cutting.py
python mst_single_linkage.py
python benchmark.py
```
