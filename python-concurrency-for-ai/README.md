# Python Concurrency for AI Workloads

Verified, runnable code from the [DadOps blog post](https://dadops.dev/blog/python-concurrency-for-ai/).

## Scripts

| Script | Description | Dependencies |
|--------|-------------|--------------|
| `gil_benchmark.py` | GIL impact: threading vs multiprocessing for CPU/I/O | None (stdlib) |
| `async_vs_threading.py` | asyncio vs threading for batch API calls | None (stdlib) |
| `tokenization_benchmark.py` | Parallel tokenization: sequential/threads/processes | None (stdlib) |
| `serialization_tax.py` | Pickle vs shared memory for large arrays | numpy |
| `hybrid_pipeline.py` | Sequential vs async-only vs hybrid RAG pipeline | None (stdlib) |
| `production_patterns.py` | Retry with backoff, semaphore, TaskGroup | None (stdlib) |

## Quick Start

```bash
pip install -r requirements.txt
python gil_benchmark.py
python async_vs_threading.py
python tokenization_benchmark.py
python serialization_tax.py
python hybrid_pipeline.py
python production_patterns.py
```

## Notes

- The GIL benchmark and tokenization benchmark are CPU-intensive — run times vary by machine.
- The async_vs_threading benchmark uses simulated delays (sleep), so timing is deterministic.
- The 500 MB shared memory benchmark requires ~1 GB free RAM.
- The production_patterns script has a 10% random failure rate — it may occasionally fail with ExceptionGroup.
