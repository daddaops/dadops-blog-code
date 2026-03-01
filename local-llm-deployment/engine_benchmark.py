"""
Benchmark harness comparing Ollama, llama.cpp, and vLLM.

From: https://dadops.dev/blog/local-llm-deployment/

Sends the same prompts to all three engines and measures
tokens/sec and latency. Requires engines running locally.

Dependencies: requests
"""

import time
import requests


def benchmark_engine(base_url: str, model: str, prompts: list[str],
                     max_tokens: int = 128) -> dict:
    """Benchmark a single engine with sequential requests."""
    results = []
    for prompt in prompts:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        start = time.perf_counter()
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload)
        elapsed = time.perf_counter() - start
        data = resp.json()

        tokens = data["usage"]["completion_tokens"]
        results.append({
            "tokens": tokens,
            "total_time": elapsed,
            "tokens_per_sec": tokens / elapsed
        })

    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_time = sum(r["total_time"] for r in results) / len(results)
    return {"avg_tok_per_sec": round(avg_tps, 1), "avg_latency": round(avg_time, 2)}


# Test prompts — mix of short and medium generation tasks
prompts = [
    "What is the capital of France? Answer in one sentence.",
    "Explain the difference between a list and a tuple in Python.",
    "Write a function that checks if a string is a palindrome.",
    "Summarize the key ideas behind MapReduce in 3 bullet points.",
    "What are the tradeoffs between SQL and NoSQL databases?",
]

engines = {
    "Ollama":    ("http://localhost:11434", "llama3.1:8b"),
    "llama.cpp": ("http://localhost:8080",  "llama3.1:8b"),
    "vLLM":      ("http://localhost:8000",  "meta-llama/Llama-3.1-8B"),
}


if __name__ == "__main__":
    print("=== Engine Benchmark ===")
    print("Requires Ollama (port 11434), llama.cpp (port 8080),")
    print("and vLLM (port 8000) running locally with the same model.")
    print()
    print(f"{'Engine':<12} {'Avg tok/s':>10} {'Avg latency':>12}")
    print("-" * 36)
    for name, (url, model) in engines.items():
        try:
            result = benchmark_engine(url, model, prompts)
            print(f"{name:<12} {result['avg_tok_per_sec']:>9.1f} {result['avg_latency']:>10.2f}s")
        except Exception as e:
            print(f"{name:<12} {'error':>10} — {e}")
