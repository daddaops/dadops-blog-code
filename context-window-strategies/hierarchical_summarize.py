"""Hierarchical summarization for very large documents.

Recursively summarizes chunks until the result fits a target token budget.
Each level compresses by roughly 5x, trading detail for breadth.

Requires: ANTHROPIC_API_KEY environment variable.
"""

import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()


def chunk_document(text, chunk_size=2000, overlap=200):
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


async def summarize_chunk(chunk, prior_context=""):
    """Summarize one chunk, with optional context from prior summaries."""
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Summarize this text concisely. "
                f"Preserve key facts, numbers, and names.\n"
                f"Prior context: {prior_context[:200]}\n\n{chunk}"
            )
        }]
    )
    return response.content[0].text


async def hierarchical_summarize(text, target_tokens=1500, level=0):
    """Recursively summarize until text fits the token budget."""
    current_tokens = int(len(text.split()) * 1.3)

    if current_tokens <= target_tokens:
        print(f"  Level {level}: {current_tokens} tokens fits budget")
        return text

    chunks = chunk_document(text, chunk_size=3000, overlap=300)
    print(f"  Level {level}: {current_tokens} tokens in {len(chunks)} chunks")

    # Summarize all chunks in parallel
    summaries = await asyncio.gather(*[
        summarize_chunk(chunk) for chunk in chunks
    ])

    compressed = "\n\n".join(summaries)
    new_tokens = int(len(compressed.split()) * 1.3)
    print(f"  Level {level}: compressed to {new_tokens} tokens "
          f"({current_tokens / max(new_tokens, 1):.1f}x)")

    # Recurse until it fits
    return await hierarchical_summarize(
        compressed, target_tokens, level + 1
    )


if __name__ == "__main__":
    print("=== Hierarchical Summarization Demo ===\n")

    # Show the compression math without API calls
    doc_sizes = [500_000, 100_000, 25_000]
    compression_ratio = 5.0

    print("Compression levels for a 500K-token document (target: 1,500 tokens):\n")
    tokens = 500_000
    level = 0
    while tokens > 1500:
        chunks = max(1, tokens // 3000)
        print(f"  Level {level}: {tokens:>8,} tokens -> {chunks} chunks")
        tokens = int(tokens / compression_ratio)
        level += 1
    print(f"  Level {level}: {tokens:>8,} tokens (fits budget)\n")

    print("To run the full pipeline with the Anthropic API:")
    print("  Set ANTHROPIC_API_KEY and call:")
    print("  asyncio.run(hierarchical_summarize(your_document, target_tokens=1500))")
    print("\nDone.")
