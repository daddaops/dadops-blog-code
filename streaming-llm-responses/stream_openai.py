"""
Stream an LLM response from OpenAI using their Python SDK.

Demonstrates the stream=True pattern with token-by-token output
and usage tracking.

Requires: OPENAI_API_KEY environment variable.

From: https://dadops.dev/blog/streaming-llm-responses/
"""

from openai import OpenAI

client = OpenAI()
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}],
    stream=True,
    stream_options={"include_usage": True},  # get token counts
)

full_response = ""
for chunk in stream:
    # The final chunk has empty choices but carries usage data
    if chunk.choices:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_response += delta

    # Usage arrives in the second-to-last chunk
    if chunk.usage:
        print(f"\n[Tokens: {chunk.usage.total_tokens}]")
