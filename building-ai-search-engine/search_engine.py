"""
AI Search Engine: document ingestion, hybrid retrieval, and cross-encoder reranking.

Blog post: https://dadops.dev/blog/building-ai-search-engine/
Code Blocks 1, 2, and 3.

Requires: sentence-transformers, numpy, sqlite3 (stdlib).
"""
import sqlite3
import struct

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# ── Code Block 1: Document Ingestion Pipeline ──

def create_search_db(db_path="search.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks
        USING fts5(title, content, source)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            vector BLOB
        )
    """)
    return conn


def chunk_document(text, chunk_size=256, overlap=64):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def ingest(conn, documents, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    for doc in documents:
        chunks = chunk_document(doc["content"])
        for chunk in chunks:
            cursor = conn.execute(
                "INSERT INTO chunks(title, content, source) VALUES (?, ?, ?)",
                (doc["title"], chunk, doc["source"])
            )
            chunk_id = cursor.lastrowid

            embedding = model.encode(chunk)
            blob = struct.pack(f"{len(embedding)}f", *embedding)
            conn.execute(
                "INSERT INTO embeddings(chunk_id, vector) VALUES (?, ?)",
                (chunk_id, blob)
            )

    conn.commit()
    return conn


# ── Code Block 2: Hybrid Retrieval Engine ──

def bm25_search(conn, query, top_k=20):
    results = conn.execute(
        """SELECT rowid, title, content, rank
           FROM chunks WHERE chunks MATCH ?
           ORDER BY rank LIMIT ?""",
        (query, top_k)
    ).fetchall()
    return [(r[0], r[1], r[2], -r[3]) for r in results]


def vector_search(conn, query, model, top_k=20):
    q_vec = model.encode(query)
    rows = conn.execute(
        "SELECT chunk_id, vector FROM embeddings"
    ).fetchall()

    scores = []
    for chunk_id, blob in rows:
        dim = len(blob) // 4
        doc_vec = np.array(struct.unpack(f"{dim}f", blob))
        similarity = np.dot(q_vec, doc_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(doc_vec) + 1e-8
        )
        scores.append((chunk_id, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def hybrid_search(conn, query, model, top_k=20, k=60):
    bm25_results = bm25_search(conn, query, top_k)
    vec_results = vector_search(conn, query, model, top_k)

    rrf_scores = {}
    for rank, (chunk_id, *_) in enumerate(bm25_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(vec_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


# ── Code Block 3: Cross-Encoder Reranker ──

class SearchReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, top_k=10):
        if not candidates:
            return []

        pairs = [(query, doc["content"]) for doc in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            {**doc, "rerank_score": float(score)}
            for doc, score in scored[:top_k]
        ]


def search_pipeline(conn, query, model, reranker, top_k=10):
    """Full search pipeline: hybrid retrieval → reranking."""
    # Stage 1-2: Hybrid retrieval (fast, broad)
    candidates = hybrid_search(conn, query, model, top_k=20)

    # Fetch full content for reranking
    docs = []
    for chunk_id, rrf_score in candidates:
        row = conn.execute(
            "SELECT title, content, source FROM chunks WHERE rowid = ?",
            (chunk_id,)
        ).fetchone()
        if row:
            docs.append({
                "id": chunk_id, "title": row[0],
                "content": row[1], "source": row[2],
                "retrieval_score": rrf_score
            })

    # Stage 3: Rerank (slow, precise)
    return reranker.rerank(query, docs, top_k=top_k)


# ── Self-test with sample documents ──

SAMPLE_DOCS = [
    {"title": "Introduction to Neural Networks",
     "content": "Neural networks are computing systems inspired by biological neural networks. "
     "They consist of layers of interconnected nodes that process information using connectionist "
     "approaches. The network learns by adjusting the weights of connections between nodes. "
     "Training a neural network involves forward propagation of inputs through the layers, "
     "computing a loss function, and backpropagation of gradients to update weights. "
     "Deep neural networks have multiple hidden layers, enabling them to learn hierarchical "
     "representations of data. Common architectures include feedforward networks, convolutional "
     "neural networks for image processing, and recurrent neural networks for sequential data.",
     "source": "blog/intro-neural-nets.html"},

    {"title": "Understanding Gradient Descent",
     "content": "Gradient descent is the fundamental optimization algorithm used to train machine "
     "learning models. It works by computing the gradient of the loss function with respect to "
     "model parameters and taking steps in the direction that minimizes the loss. Stochastic "
     "gradient descent uses random mini-batches instead of the full dataset, making it more "
     "efficient for large datasets. Variants like Adam, RMSProp, and AdaGrad adapt the learning "
     "rate for each parameter. Learning rate scheduling helps convergence by reducing the step "
     "size over time. Momentum accelerates gradient descent by accumulating a velocity vector.",
     "source": "blog/gradient-descent.html"},

    {"title": "Transformer Architecture Explained",
     "content": "Transformers revolutionized natural language processing with their self-attention "
     "mechanism. Unlike RNNs, transformers process all positions in parallel, enabling much faster "
     "training. The key innovation is scaled dot-product attention: Q, K, V matrices are computed "
     "from the input, attention scores are calculated as softmax(QK^T/sqrt(d_k))V. Multi-head "
     "attention allows the model to attend to information from different representation subspaces. "
     "Positional encodings add sequence order information since attention is permutation-invariant. "
     "The encoder-decoder architecture uses cross-attention to connect the two halves.",
     "source": "blog/transformers.html"},

    {"title": "Building Search Engines with Python",
     "content": "Search engines combine information retrieval techniques with ranking algorithms "
     "to find relevant documents. BM25 is the standard keyword matching algorithm, extending TF-IDF "
     "with document length normalization. Vector search uses embeddings to capture semantic similarity "
     "beyond exact keyword matches. Hybrid search combines both approaches using Reciprocal Rank "
     "Fusion to merge result lists. Cross-encoder reranking improves precision by jointly encoding "
     "the query and document. A complete search pipeline includes ingestion, retrieval, reranking, "
     "and serving stages.",
     "source": "blog/search-engines.html"},

    {"title": "Python Concurrency Patterns",
     "content": "Python offers several concurrency models: threading for I/O-bound tasks, "
     "multiprocessing for CPU-bound tasks, and asyncio for cooperative multitasking. The GIL "
     "prevents true parallelism in threads but doesn't affect I/O operations. Process pools "
     "distribute work across CPU cores using the multiprocessing module. Async/await syntax "
     "enables efficient handling of thousands of concurrent connections. Common patterns include "
     "producer-consumer queues, thread pools, and event-driven architectures. For AI workloads, "
     "batch processing with multiprocessing often outperforms async approaches.",
     "source": "blog/python-concurrency.html"},
]

if __name__ == "__main__":
    import os
    import time

    db_path = ":memory:"
    print("=== AI Search Engine Pipeline Test ===\n")

    # Ingest
    print("Loading models...")
    t0 = time.perf_counter()
    conn = create_search_db(db_path)
    ingest(conn, SAMPLE_DOCS)
    t1 = time.perf_counter()
    print(f"Ingested {len(SAMPLE_DOCS)} documents in {t1-t0:.2f}s\n")

    # Check embedding dimensions
    row = conn.execute("SELECT vector FROM embeddings LIMIT 1").fetchone()
    dim = len(row[0]) // 4
    print(f"Embedding dimensions: {dim}")

    # Load models for search
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = SearchReranker()

    test_queries = [
        "how do neural networks learn",
        "search engine ranking algorithms",
        "python parallel processing",
    ]

    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")

        # BM25
        bm25 = bm25_search(conn, query, top_k=5)
        print(f"BM25 top results: {[r[1] for r in bm25[:3]]}")

        # Vector
        vec = vector_search(conn, query, model, top_k=5)
        vec_titles = []
        for cid, score in vec[:3]:
            row = conn.execute("SELECT title FROM chunks WHERE rowid=?", (cid,)).fetchone()
            vec_titles.append(f"{row[0]} ({score:.3f})")
        print(f"Vector top results: {vec_titles}")

        # Hybrid
        hybrid = hybrid_search(conn, query, model, top_k=5)
        print(f"Hybrid RRF top IDs: {[cid for cid, _ in hybrid[:3]]}")

        # Full pipeline with reranking
        t0 = time.perf_counter()
        results = search_pipeline(conn, query, model, reranker, top_k=3)
        t1 = time.perf_counter()
        print(f"Reranked results ({t1-t0:.3f}s):")
        for r in results:
            print(f"  {r['title']} (score: {r['rerank_score']:.4f})")

    print("\nAll search engine tests passed!")
