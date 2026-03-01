# LLM Function Calling Done Right

Verified, runnable code from the [LLM Function Calling](https://dadops.dev/blog/llm-function-calling/) blog post.

## Scripts

| Script | Description | API Key Required? |
|--------|-------------|-------------------|
| `parse_tool_call.py` | Regex-based tool call parsing (pre-API approach) | No |
| `tool_implementations.py` | Shared tool functions + OpenAI/Anthropic schemas | No |
| `openai_agent.py` | OpenAI native function calling agent loop | Yes (OPENAI_API_KEY) |
| `anthropic_agent.py` | Anthropic native function calling agent loop | Yes (ANTHROPIC_API_KEY) |
| `verify_tools.py` | Verification suite — tests all logic without API keys | No |

## Quick Start

```bash
pip install -r requirements.txt

# Run verification (no API key needed)
python verify_tools.py

# Run with OpenAI (requires API key)
export OPENAI_API_KEY=sk-...
python openai_agent.py

# Run with Anthropic (requires API key)
export ANTHROPIC_API_KEY=sk-ant-...
python anthropic_agent.py
```
