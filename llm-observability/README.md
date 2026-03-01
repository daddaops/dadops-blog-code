# LLM Observability — Verified Code

Runnable code from the DadOps blog post:
**[LLM Observability in Production: Tracing, Logging, and Monitoring AI Systems](https://daddaops.com/blog/llm-observability/)**

## Scripts

| Script | Blog Section | Description |
|--------|-------------|-------------|
| `llm_call_logger.py` | Code Block 1-2 | Structured logging decorator for async LLM calls with cost calculation |
| `cost_tracker.py` | Code Block 3 | Real-time cost tracker with anomaly detection (z-score) and daily reports |
| `latency_analyzer.py` | Code Block 4 | Latency segmentation by model/bucket, percentiles, regression detection |
| `quality_monitor.py` | Code Block 5 | 3-signal quality health score (length stability, embeddings, judge) |
| `trace_builder.py` | Code Block 6 | Distributed tracing with context manager spans and tree summary |
| `adaptive_alerts.py` | Code Block 7 | Adaptive alerting with time-of-week awareness and floor/ceiling metrics |
| `verify_observability.py` | All | Verification suite testing all logic and blog claims |

## Running

All scripts use only the Python standard library — no `pip install` needed:

```bash
python3 llm_call_logger.py
python3 cost_tracker.py
python3 latency_analyzer.py
python3 quality_monitor.py
python3 trace_builder.py
python3 adaptive_alerts.py
python3 verify_observability.py
```

## Dependencies

None — all code uses only the Python standard library (`time`, `json`, `hashlib`, `uuid`, `dataclasses`, `functools`, `logging`, `collections`, `datetime`, `statistics`, `enum`, `contextlib`).

The logging decorator (`llm_call_logger.py`) assumes the wrapped function returns an OpenAI-compatible response object, but does not import `openai` itself. The test uses mock objects.
