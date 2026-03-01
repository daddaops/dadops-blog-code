# Local LLM Deployment — Verified Code

Runnable code from [Running LLMs on Your Own Machine](https://dadops.dev/blog/local-llm-deployment/) on DadOps.

## Scripts

| Script | Description | Requirements |
|--------|-------------|-------------|
| `vram_estimator.py` | VRAM estimation function + table output | None (stdlib) |
| `ollama_client.py` | OpenAI-compatible client for Ollama | openai, Ollama running |
| `vllm_benchmark.py` | Async concurrent throughput benchmark | openai, vLLM running |
| `engine_benchmark.py` | Compare Ollama/llama.cpp/vLLM | requests, all engines running |
| `verify_local_llm.py` | Full verification of VRAM math + blog claims | None (stdlib) |

## Quick Start

```bash
# No dependencies needed for VRAM estimation
python3 vram_estimator.py           # Print VRAM table
python3 verify_local_llm.py         # Run full verification suite
```

## With Local Engines

```bash
pip install openai requests
ollama pull llama3.2 && ollama serve &
python3 ollama_client.py
```
