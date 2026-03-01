# Streaming LLM Responses — Verified Code

Runnable code from the DadOps blog post: [Streaming LLM Responses: Server-Sent Events, Chunked Transfer, and the UX of Waiting](https://daddaops.com/blog/streaming-llm-responses/)

## Scripts

| Script | Description | API Key Required? |
|--------|-------------|-------------------|
| `stream_openai.py` | Stream from OpenAI with SDK | Yes (OPENAI_API_KEY) |
| `stream_anthropic.py` | Stream from Anthropic with SDK | Yes (ANTHROPIC_API_KEY) |
| `stream_raw_httpx.py` | Raw HTTP SSE parsing with httpx | Yes (OPENAI_API_KEY) |
| `fastapi_relay.py` | FastAPI streaming relay server | Yes (OPENAI_API_KEY) |

## Quick Start

```bash
pip install -r requirements.txt

# OpenAI streaming
export OPENAI_API_KEY=your-key-here
python stream_openai.py

# Anthropic streaming
export ANTHROPIC_API_KEY=your-key-here
python stream_anthropic.py

# Raw HTTP streaming
python stream_raw_httpx.py

# FastAPI relay server
uvicorn fastapi_relay:app --reload
```

## Notes

- All scripts require external API keys to run
- The JavaScript code blocks in the blog post (EventSource, fetch+ReadableStream, requestAnimationFrame) are browser-side patterns — not extracted as standalone scripts
- The latency comparison table uses data from Artificial Analysis benchmarks
- The FastAPI relay demonstrates the streaming proxy pattern with client disconnect detection
