"""
Embedding and retrieval for RAG pipelines.

Demonstrates how to embed text chunks using sentence-transformers
and retrieve the most relevant chunks for a given query using
cosine similarity (dot product on normalized vectors).

From: https://dadops.dev/blog/rag-from-scratch/
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from chunker import chunk_text


def retrieve(query, chunks, embeddings, model, k=3):
    """Find the k most relevant chunks for a query."""
    # Embed the query with the same model
    query_vec = model.encode([query], normalize_embeddings=True)

    # Dot product against all chunk embeddings (cosine sim for normalized vecs)
    scores = (embeddings @ query_vec.T).squeeze()

    # Get top-k indices
    top_k = np.argsort(-scores)[:k]

    return [(chunks[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    # Sample document
    sample = """Solar panels convert sunlight into electricity through the
photovoltaic effect. When photons hit silicon cells, they knock electrons
loose, creating an electrical current.

Installation requires careful roof assessment. South-facing roofs with
15-40 degree pitch are ideal in the northern hemisphere. Shading from
trees or neighboring buildings can reduce output by 10-25%.

A typical residential system is 6-10 kW, requiring 15-25 panels. At
average US electricity rates, payback period is 6-10 years. Federal tax
credits currently cover 30% of installation costs."""

    # Step 1: Chunk
    chunks = chunk_text(sample, max_chars=300, overlap=60)
    print(f"Created {len(chunks)} chunks\n")

    # Step 2: Embed
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, normalize_embeddings=True)
    print(f"Embedded {len(chunks)} chunks → {embeddings.shape}\n")

    # Step 3: Retrieve
    results = retrieve(
        "How long until solar panels pay for themselves?",
        chunks, embeddings, model, k=2
    )

    print("Query: 'How long until solar panels pay for themselves?'\n")
    for chunk, score in results:
        print(f"Score: {score:.3f}")
        print(chunk[:100] + "...")
        print()
