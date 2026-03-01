"""
Stream an LLM response from Anthropic using their Python SDK.

Demonstrates the typed event stream with text_stream iteration
and post-stream usage tracking.

Requires: ANTHROPIC_API_KEY environment variable.

From: https://dadops.dev/blog/streaming-llm-responses/
"""

import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# After the stream closes, get full message with usage
message = stream.get_final_message()
print(f"\n[Tokens: {message.usage.input_tokens + message.usage.output_tokens}]")
