# RAG from Scratch — Verified Code

Runnable code from the DadOps blog post: [Building a RAG Pipeline from Scratch](https://dadops.dev/blog/rag-from-scratch/)

## Scripts

| Script | Description | API Key Required? |
|--------|-------------|-------------------|
| `chunker.py` | Recursive text chunker with overlap | No |
| `embed_and_retrieve.py` | Embed chunks + cosine similarity retrieval | No |
| `rag_pipeline.py` | Full RAG pipeline with LLM generation | Yes (ANTHROPIC_API_KEY) |

## Quick Start

```bash
pip install -r requirements.txt

# Run chunker demo (no dependencies beyond stdlib)
python chunker.py

# Run embedding + retrieval demo (needs sentence-transformers)
python embed_and_retrieve.py

# Run full pipeline (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your-key-here
python rag_pipeline.py
```

## Notes

- `chunker.py` is pure Python with no external dependencies
- `embed_and_retrieve.py` uses `all-MiniLM-L6-v2` (downloaded automatically on first run)
- `rag_pipeline.py` requires an Anthropic API key for the generation step; the chunking and retrieval parts work without one
