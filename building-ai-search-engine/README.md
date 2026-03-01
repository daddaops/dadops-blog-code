# Building an AI Search Engine

Verified, runnable code from the DadOps blog post:
[Building an AI Search Engine](https://dadops.dev/blog/building-ai-search-engine/)

## Scripts

| Script | Code Blocks | What it does |
|--------|------------|-------------|
| `search_engine.py` | 1, 2, 3 | Full pipeline: ingestion, hybrid search (BM25 + vector + RRF), cross-encoder reranking |
| `search_api.py` | 4 | FastAPI endpoint with snippet highlighting (runs highlight tests standalone) |
| `evaluation.py` | 8 | P@K, R@K, NDCG@K metrics — evaluates BM25 vs Hybrid vs Reranked on synthetic corpus |

## Usage

```bash
pip install -r requirements.txt

# Core pipeline test (ingests 5 sample docs, runs queries through all stages):
python search_engine.py

# Snippet highlighting + Pydantic model tests (no server needed):
python search_api.py

# Full evaluation — compares BM25, Hybrid, and Reranked P@5 on 30 docs / 20 queries:
python evaluation.py
```

## Blog Claims vs Reality

The blog claims P@5 scores of 0.52 → 0.64 → 0.78 for BM25 → Hybrid → Reranked on 200 docs / 20 queries.
Our evaluation uses 30 docs / 20 queries — a smaller corpus, but the directional improvement
(Hybrid > BM25, Reranked > Hybrid) is verified.

## Notes

- Models downloaded on first run: `all-MiniLM-L6-v2` (bi-encoder) + `ms-marco-MiniLM-L-6-v2` (cross-encoder)
- Code Blocks 5-7 (curl example, JSON response, JS frontend) are reference/presentation — not extracted as scripts
