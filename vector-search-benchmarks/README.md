# Vector Search at Small Scale: pgvector vs FAISS vs Brute Force NumPy

Verified, runnable code from the [DadOps blog post](https://dadops.dev/blog/vector-search-benchmarks/).

## Scripts

| Script | Description | Runtime |
|--------|-------------|---------|
| `demo_numpy.py` | NumPy brute-force cosine search demo | ~5 seconds |
| `demo_faiss.py` | FAISS Flat/IVF/HNSW index demo | ~30 seconds |
| `benchmark_all.py` | Full benchmark: 4 dims × 3 sizes × 4-6 methods | ~30-60 minutes |

## Quick Start

```bash
pip install -r requirements.txt
python demo_numpy.py       # NumPy only (no extra deps)
python demo_faiss.py       # Needs faiss-cpu
python benchmark_all.py    # Full benchmark (NumPy + FAISS)
```

## pgvector Benchmarks

pgvector benchmarks require a running PostgreSQL instance with pgvector:

```bash
docker run -d --name pgvec -e POSTGRES_PASSWORD=secret -p 5432:5432 pgvector/pgvector:pg16
PGVECTOR_ENABLED=1 python benchmark_all.py
```

## Test Matrix

- **Dimensions:** 128, 384, 768, 1536
- **Dataset sizes:** 10K, 50K, 100K vectors
- **Methods:** NumPy brute-force, FAISS Flat, FAISS IVF, FAISS HNSW, pgvector HNSW, pgvector IVF
- **Queries:** 100 random unit vectors per configuration
- **Metric:** Cosine similarity (via inner product on unit vectors)
