"""Vector Search using FAISS + sentence-transformers

Semantic search using dense embeddings. Transforms documents and queries into
384-dimensional vectors where semantic similarity maps to geometric proximity.

Requires: faiss-cpu, sentence-transformers

Blog post: https://dadops.co/blog/hybrid-search-benchmarks/
Code Block 2 from the blog.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class VectorSearch:
    """Semantic search using FAISS + sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_ids = []

    def index_documents(self, documents: List[Tuple[str, str]]):
        """Embed and index a list of (doc_id, content) tuples."""
        self.doc_ids = [doc_id for doc_id, _ in documents]
        texts = [content for _, content in documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True,
                                       show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine on normalized vecs
        self.index.add(embeddings)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, cosine_similarity) pairs."""
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, indices = self.index.search(q_emb, k)
        return [(self.doc_ids[i], float(scores[0][j]))
                for j, i in enumerate(indices[0]) if i != -1]


if __name__ == "__main__":
    print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    vec = VectorSearch()

    docs = [
        ("doc1", "PostgreSQL query optimization and index tuning for faster databases"),
        ("doc2", "Error E-4012: connection refused when connecting to the database server"),
        ("doc3", "How to speed up your database with proper indexing strategies"),
        ("doc4", "Python asyncio rate limiting patterns for API calls"),
        ("doc5", "Kubernetes pod OOMKilled troubleshooting guide"),
        ("doc6", "Machine learning model deployment best practices"),
        ("doc7", "Database connection pooling with setMaxPoolSize parameter"),
        ("doc8", "SSL certificate configuration and HTTPS setup guide"),
        ("doc9", "Caching strategies for reducing database load"),
        ("doc10", "Docker container memory limits and resource management"),
    ]

    print("Indexing documents...")
    vec.index_documents(docs)

    print("\n=== Vector Search Tests ===\n")

    # Semantic search (vector should excel)
    results = vec.search("how to make queries faster", k=5)
    print("Query: 'how to make queries faster'")
    for doc_id, score in results:
        print(f"  {doc_id}: cosine={score:.4f}")

    print()

    # Exact keyword search (vector should struggle)
    results = vec.search("error E-4012 connection refused", k=5)
    print("Query: 'error E-4012 connection refused'")
    for doc_id, score in results:
        print(f"  {doc_id}: cosine={score:.4f}")

    print()

    # Another semantic search
    results = vec.search("preventing data loss during crashes", k=5)
    print("Query: 'preventing data loss during crashes'")
    for doc_id, score in results:
        print(f"  {doc_id}: cosine={score:.4f}")
