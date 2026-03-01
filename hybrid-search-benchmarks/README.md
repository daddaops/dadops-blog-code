# Hybrid Search Benchmarks — Verified Code

Runnable code from the DadOps blog post: [Hybrid Search Benchmarks: BM25 + Vector Search vs Either Alone](https://dadops.co/blog/hybrid-search-benchmarks/)

## Scripts

- `bm25_search.py` — BM25 search using SQLite FTS5 (pure Python, no GPU needed)
- `vector_search.py` — Semantic search using FAISS + sentence-transformers
- `hybrid_search.py` — RRF and WLC fusion combining BM25 + vector results
- `benchmark.py` — End-to-end benchmark: generates test corpus, runs all methods, measures latency and quality

## Quick Start

```bash
pip install -r requirements.txt
python benchmark.py
```

## Dependencies

- `faiss-cpu` — Facebook AI Similarity Search (CPU version)
- `sentence-transformers` — Sentence embedding models
- `numpy` — Numerical operations
