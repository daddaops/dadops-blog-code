# Profiling Python AI Code: Finding the Bottleneck Before You Optimize

Verified, runnable code from the [DadOps blog post](https://dadops.dev/blog/profiling-python-ai-code/).

## Scripts

| Script | Description | Dependencies |
|--------|-------------|--------------|
| `cprofile_tokenization.py` | cProfile on batch tokenization (5K docs) | None (stdlib) |
| `benchmark_overhead.py` | Head-to-head profiler overhead comparison | numpy; optional: py-spy, scalene |
| `benchmarks/*.py` | Individual workload scripts for the harness | numpy |
| `shared_memory_vs_pickle.py` | Pickle vs shared memory serialization demo | numpy |
| `memory_profiling.py` | tracemalloc leak detection demo | numpy |

## Quick Start

```bash
pip install -r requirements.txt
python cprofile_tokenization.py
python benchmark_overhead.py
python shared_memory_vs_pickle.py
python memory_profiling.py
```

## Notes

- Code blocks 2 (py-spy bash commands) and 3 (Scalene annotated pipeline) are illustrative examples
  showing tool usage patterns and output format — they require external tools and real ML models.
  The key patterns are demonstrated in the runnable scripts above with mock models.
- The benchmark harness runs cProfile by default. Install `py-spy` and `scalene` for full comparison.
