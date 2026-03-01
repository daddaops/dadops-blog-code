# Building AI Agents with Tool Use: From Chat to Action

Verified, runnable code from the DadOps blog post:
[Building AI Agents with Tool Use](https://dadops.dev/blog/building-ai-agents/)

## Scripts

- `tool_functions.py` — Standalone tool functions (search_files, read_file, calculate, truncate_result) with self-tests. NO API key needed.
- `agent_openai.py` — OpenAI Agent class with weather tool demo and research assistant. Requires `OPENAI_API_KEY`.
- `agent_anthropic.py` — Anthropic ClaudeAgent class with weather tool demo. Requires `ANTHROPIC_API_KEY`.

## Usage

```bash
pip install -r requirements.txt

# Runs without API keys:
python tool_functions.py

# Requires API keys:
# OPENAI_API_KEY=sk-... python agent_openai.py
# ANTHROPIC_API_KEY=sk-... python agent_anthropic.py
```
