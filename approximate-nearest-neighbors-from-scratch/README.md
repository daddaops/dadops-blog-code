# Approximate Nearest Neighbors from Scratch

Verified, runnable code from the DadOps blog post:
[Approximate Nearest Neighbors from Scratch](https://daddaops.com/blog/approximate-nearest-neighbors-from-scratch/)

## Scripts

| Script | Description |
|--------|-------------|
| `lsh.py` | Random Hyperplane LSH — multi-table hashing for cosine similarity |
| `nsw.py` | Navigable Small World graph — proximity graph with O(log n) search |
| `hnsw.py` | Hierarchical NSW — multi-layer skip-list graph structure |
| `product_quantization.py` | Product Quantization with Asymmetric Distance Computation |
| `benchmark_ann.py` | Recall vs QPS benchmark sweeping ef_search on NSW |

## Usage

```bash
pip install -r requirements.txt
python3 lsh.py
python3 nsw.py
python3 hnsw.py
python3 product_quantization.py
python3 benchmark_ann.py
```

All scripts use seeded RNGs for reproducible output.
