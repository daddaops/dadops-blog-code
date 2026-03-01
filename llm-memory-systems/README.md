# LLM Memory Systems

Verified, runnable code from the [LLM Memory Systems](https://dadops.dev/blog/llm-memory-systems/) blog post.

## Scripts

| Script | Description | Dependencies |
|--------|-------------|-------------|
| `conversation_buffer.py` | Simple buffer — stores every message | None |
| `sliding_window.py` | Keeps recent + important older messages | None |
| `summary_memory.py` | Compresses conversation into running summaries | None |
| `entity_memory.py` | Structured entity facts with conflict resolution | None |
| `semantic_memory.py` | Vector-based long-term memory with time decay | numpy |
| `hybrid_manager.py` | Combines all memory types into one manager | numpy |
| `verify_memory.py` | Verification suite — tests all implementations | numpy |

## Quick Start

```bash
pip install -r requirements.txt
python verify_memory.py
```

No API keys required — all LLM calls are simulated.
