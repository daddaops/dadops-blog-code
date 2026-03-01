"""
OpenAI-compatible client for Ollama (local LLM server).

From: https://dadops.dev/blog/local-llm-deployment/

Demonstrates calling a local Ollama server using the same OpenAI
library used for cloud API calls. Requires Ollama running locally
with a model pulled (e.g., `ollama pull llama3.2`).

Dependencies: openai
"""

from openai import OpenAI

# Same library, same interface — just point to localhost
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # Ollama doesn't require auth
)


def chat_local(prompt, model="llama3.2", temperature=0.7, max_tokens=256):
    """Send a chat request to local Ollama server."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("=== Ollama Client ===")
    print("Requires Ollama running locally: `ollama serve`")
    print("And a model pulled: `ollama pull llama3.2`")
    print()
    try:
        result = chat_local("Explain KV caching in one paragraph.")
        print(f"Response:\n{result}")
    except Exception as e:
        print(f"Connection failed (Ollama not running?): {e}")
