"""
Consume an OpenAI streaming response with raw HTTP using httpx.

Shows what the SDK hides: SSE message parsing, buffer management,
and the data: prefix protocol.

Requires: OPENAI_API_KEY environment variable (passed as argument or env var).

From: https://dadops.dev/blog/streaming-llm-responses/
"""

import httpx
import json
import os


def stream_raw(prompt: str, api_key: str):
    """Consume OpenAI streaming response with raw HTTP."""
    with httpx.stream(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
    ) as response:
        buffer = ""
        for raw_bytes in response.iter_bytes():
            buffer += raw_bytes.decode("utf-8")
            # SSE messages are separated by double newlines
            while "\n\n" in buffer:
                message, buffer = buffer.split("\n\n", 1)
                for line in message.split("\n"):
                    if line.startswith("data: "):
                        payload = line[6:]  # strip "data: " prefix
                        if payload == "[DONE]":
                            return
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    stream_raw("Explain streaming in 3 sentences.", api_key)
    print()  # trailing newline
