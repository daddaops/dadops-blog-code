# Structured Output from LLMs — Verified Code

Runnable code from the DadOps blog post: [Structured Output from LLMs: Getting JSON Every Time](https://daddaops.com/blog/structured-output-from-llms/)

## Scripts

| Script | Description | API Key Required? |
|--------|-------------|-------------------|
| `json_mode_demo.py` | OpenAI JSON mode, JSON Schema mode, Anthropic prefill | Yes (both keys) |
| `tool_use_demo.py` | Anthropic tool use + OpenAI function calling | Yes (both keys) |
| `instructor_demo.py` | Pydantic + instructor with both providers | Yes (both keys) |
| `validation_demo.py` | Semantic validation with Receipt model | **No** (Pydantic only) |

## Quick Start

```bash
pip install -r requirements.txt

# Validation demo runs without any API keys
python validation_demo.py

# API-dependent demos
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
python json_mode_demo.py
python tool_use_demo.py
python instructor_demo.py
```

## Notes

- `validation_demo.py` is the only script that runs without API keys — it tests Pydantic model/field validators with synthetic receipt data
- `instructor_demo.py` also tests the Pydantic model validation locally before attempting API calls
- The temperature × structure code block in the blog is illustrative (simulated data), not a runnable benchmark
- The failure modes code block is a comment-only illustration of common problems
