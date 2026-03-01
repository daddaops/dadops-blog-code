"""RAG pipeline workload: mixed CPU + I/O simulation."""
import json
import hashlib
import tempfile
import os

def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def embed_chunk(chunk):
    """Simulate embedding via hashing (CPU-bound)."""
    h = hashlib.sha256(chunk.encode()).hexdigest()
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]

def search_index(query_vec, index, top_k=5):
    """Simulate vector search (CPU-bound)."""
    scores = []
    for i, vec in enumerate(index):
        score = sum(a * b for a, b in zip(query_vec, vec))
        scores.append((score, i))
    scores.sort(reverse=True)
    return scores[:top_k]

if __name__ == "__main__":
    # Generate documents
    docs = [f"Document {i} discusses topic {i % 10} in detail " * 50
            for i in range(500)]

    # Chunk and embed
    all_chunks = []
    index = []
    for doc in docs:
        chunks = chunk_text(doc)
        for chunk in chunks:
            all_chunks.append(chunk)
            index.append(embed_chunk(chunk))

    # Simulate queries
    queries = [f"query about topic {i}" for i in range(20)]
    for q in queries:
        q_vec = embed_chunk(q)
        results = search_index(q_vec, index)

    # Write results to temp file (I/O)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        tmp_path = f.name
        for i, chunk in enumerate(all_chunks[:100]):
            json.dump({"id": i, "text": chunk[:100]}, f)
            f.write("\n")
    os.unlink(tmp_path)
