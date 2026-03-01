# LLM Model Routing

Verified, runnable code from the [LLM Model Routing](https://dadops.dev/blog/llm-model-routing/) blog post.

## Scripts

| Script | Description | Dependencies | API Key? |
|--------|-------------|-------------|----------|
| `heuristic_router.py` | Keyword-based routing (3 tiers) | None | No |
| `embedding_router.py` | Sentence embedding + logistic regression | sentence-transformers, sklearn | No |
| `judge_router.py` | GPT-4o-mini as complexity classifier | openai | **Yes** |
| `cascade_router.py` | Try cheapest first, escalate on failure | None | No |
| `evaluation.py` | Router benchmarking framework | None | No |
| `production_router.py` | Circuit breaker + provider fallbacks | None | No |
| `verify_routing.py` | Verification suite (no API/ML needed) | None | No |

## Quick Start

```bash
pip install -r requirements.txt

# Run verification (no API key or ML model needed)
python verify_routing.py

# Run embedding router (downloads ~80MB model on first run)
python embedding_router.py

# Run judge router (requires API key)
export OPENAI_API_KEY=sk-...
python judge_router.py
```
