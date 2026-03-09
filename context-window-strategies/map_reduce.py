"""Map-reduce strategy for cross-document reasoning.

Processes multiple documents in parallel (map phase), extracting relevant
information from each, then synthesizes a final answer (reduce phase).

Requires: ANTHROPIC_API_KEY environment variable.
"""

import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()


async def map_document(doc, question, doc_id):
    """Map phase: extract relevant info from one document."""
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": (
                f"Document {doc_id}:\n{doc}\n\n"
                f"Question: {question}\n\n"
                f"Extract all relevant facts, numbers, and quotes."
            )
        }]
    )
    return {"doc_id": doc_id, "extraction": response.content[0].text}


async def reduce_results(extractions, question):
    """Reduce phase: synthesize extractions into a final answer."""
    combined = "\n\n".join(
        f"--- Document {e['doc_id']} ---\n{e['extraction']}"
        for e in extractions
    )
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": (
                f"Extractions from {len(extractions)} documents:\n\n"
                f"{combined}\n\nQuestion: {question}\n\n"
                f"Synthesize a comprehensive answer. "
                f"Note contradictions. Cite document numbers."
            )
        }]
    )
    return response.content[0].text


async def map_reduce(documents, question):
    """Process multiple documents with parallel map, then reduce."""
    map_tasks = [
        map_document(doc, question, i + 1)
        for i, doc in enumerate(documents)
    ]
    extractions = await asyncio.gather(*map_tasks)
    return await reduce_results(extractions, question)


if __name__ == "__main__":
    print("=== Map-Reduce Demo ===\n")
    print("This script requires an Anthropic API key to run the full pipeline.")
    print("Set ANTHROPIC_API_KEY and call:")
    print("  asyncio.run(map_reduce([doc1, doc2, ...], 'your question'))")

    # Demo the structure without API calls
    sample_docs = [
        "Q1 2024 Earnings: Revenue was $50M, up 12% YoY. Net income $8M.",
        "Q2 2024 Earnings: Revenue was $55M, up 15% YoY. Net income $10M.",
        "Q3 2024 Earnings: Revenue was $48M, down 5% YoY. Net income $6M.",
    ]
    question = "What was the revenue trend across quarters?"

    print(f"\nSample documents: {len(sample_docs)}")
    for i, doc in enumerate(sample_docs):
        print(f"  Doc {i+1}: {doc[:60]}...")
    print(f"Question: {question}")
    print("\nDone.")
