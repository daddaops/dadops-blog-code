# Building Multimodal AI Apps

Verified code from the DadOps blog post: [Building Multimodal AI Apps](https://dadops.dev/blog/building-multimodal-ai-apps/)

## Scripts

| Script | Blog Blocks | API Required | Self-Tests |
|--------|-------------|-------------|------------|
| `document_extraction.py` | 1, 2 | OpenAI (GPT-4o) | Yes (Pydantic models, merge logic) |
| `chart_analysis.py` | 3 | Anthropic (Claude) | No |
| `dashboard_monitor.py` | 4 | Anthropic + Chromium | No |
| `batch_processing.py` | 5 | OpenAI (GPT-4o-mini) | No |
| `image_comparison.py` | 6 | OpenAI (GPT-4o) | No |
| `production_orchestrator.py` | 7 | OpenAI/Anthropic | Yes (token estimation, preprocessing, cost math) |

## Running

```bash
pip install -r requirements.txt

# Scripts with self-tests (no API key needed):
python document_extraction.py
python production_orchestrator.py

# API-dependent scripts (set keys first):
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
```

## Notes

- Most scripts require API keys (OpenAI and/or Anthropic)
- `production_orchestrator.py` and `document_extraction.py` have testable pure-computation
  components (token estimation, image preprocessing, Pydantic validation, page merging)
- Token estimation verified: 1920x1080 at high detail = 1,105 tokens (matches blog claim)
