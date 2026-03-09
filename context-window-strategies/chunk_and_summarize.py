"""Chunk-and-summarize strategy for compressing long documents.

Splits a document into overlapping chunks, summarizes each with an LLM,
and concatenates the summaries into a compressed context.

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


async def chunk_and_summarize(document, chunk_size=2000, overlap=200):
    """Compress a long document by chunking and summarizing."""
    chunks = chunk_document(document, chunk_size, overlap)
    summaries = []

    for i, chunk in enumerate(chunks):
        context = summaries[-1] if summaries else ""
        summary = await summarize_chunk(chunk, context)
        summaries.append(summary)

    compressed = "\n\n".join(summaries)
    ratio = len(document.split()) / max(len(compressed.split()), 1)
    print(f"Compressed {len(chunks)} chunks: {ratio:.1f}x reduction")
    return compressed


if __name__ == "__main__":
    # Demo: chunk a sample document (chunking only, no API call)
    sample_text = " ".join(["word"] * 5000)
    print("=== Chunk-and-Summarize Demo ===\n")

    chunks = chunk_document(sample_text, chunk_size=2000, overlap=200)
    print(f"Document: {len(sample_text.split())} words")
    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk.split())} words")

    print("\nTo run the full summarization pipeline with the Anthropic API:")
    print("  Set ANTHROPIC_API_KEY and call:")
    print("  asyncio.run(chunk_and_summarize(your_document))")
    print("\nDone.")
