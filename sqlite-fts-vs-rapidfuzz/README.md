# SQLite FTS5 vs rapidfuzz: Fuzzy Search Showdown

Verified, runnable code from the DadOps blog post:
**[SQLite FTS5 vs rapidfuzz](https://dadops.dev/blog/sqlite-fts-vs-rapidfuzz/)**

## Scripts

| Script | Description |
|--------|-------------|
| `demo_rapidfuzz.py` | Basic rapidfuzz usage demo (pairwise similarity, search) |
| `benchmark_all.py` | Full benchmark pipeline: 500K dataset, FTS5 setup, speed benchmarks, hybrid search, batch benchmarks |

## Run

```bash
pip install -r requirements.txt
python3 demo_rapidfuzz.py
python3 benchmark_all.py 2>&1 | tee output/benchmark_all.log
```

**Note:** `benchmark_all.py` takes 15-30 minutes to run due to rapidfuzz linear scans on 500K records (especially the 1000-query batch benchmarks).

## Dependencies

- `rapidfuzz>=3.9` — fuzzy string matching (C++ core)
- `sqlite3` — built into Python standard library (FTS5 support required)
