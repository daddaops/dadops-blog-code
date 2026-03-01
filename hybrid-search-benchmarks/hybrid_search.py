"""Hybrid Search — RRF and WLC Fusion

Combines BM25 keyword search with vector semantic search using two fusion
strategies: Reciprocal Rank Fusion (RRF) and Weighted Linear Combination (WLC).

Requires: bm25_search.py and vector_search.py in the same directory.

Blog post: https://dadops.co/blog/hybrid-search-benchmarks/
Code Block 3 from the blog.
"""

from typing import List, Tuple, Dict
from collections import defaultdict
from bm25_search import BM25Search
from vector_search import VectorSearch


class HybridSearch:
    """Hybrid search combining BM25 and vector search with RRF or WLC fusion."""

    def __init__(self, bm25: 'BM25Search', vector: 'VectorSearch'):
        self.bm25 = bm25
        self.vector = vector

    def search_rrf(self, query: str, k: int = 10,
                   rrf_k: int = 60, fetch: int = 50) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion — rank-based, no score normalization needed."""
        bm25_results = self.bm25.search(query, k=fetch)
        vec_results = self.vector.search(query, k=fetch)

        rrf_scores: Dict[str, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
        for rank, (doc_id, _) in enumerate(vec_results):
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def search_wlc(self, query: str, k: int = 10,
                   alpha: float = 0.5, fetch: int = 50) -> List[Tuple[str, float]]:
        """Weighted Linear Combination — score-based with min-max normalization."""
        bm25_results = self.bm25.search(query, k=fetch)
        vec_results = self.vector.search(query, k=fetch)

        def normalize(results):
            if not results:
                return {}
            scores = [s for _, s in results]
            lo, hi = min(scores), max(scores)
            span = hi - lo if hi != lo else 1.0
            return {doc_id: (score - lo) / span for doc_id, score in results}

        bm25_norm = normalize(bm25_results)
        vec_norm = normalize(vec_results)

        all_ids = set(bm25_norm) | set(vec_norm)
        combined = {}
        for doc_id in all_ids:
            b = bm25_norm.get(doc_id, 0.0)
            v = vec_norm.get(doc_id, 0.0)
            combined[doc_id] = alpha * b + (1 - alpha) * v

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    print("Setting up hybrid search pipeline...")

    # Create both search engines
    bm25 = BM25Search()
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

    bm25.index(docs)
    vec.index_documents(docs)

    hybrid = HybridSearch(bm25, vec)

    print("\n=== Hybrid Search Comparison ===\n")

    queries = [
        ("error E-4012 connection refused", "exact keyword"),
        ("how to make queries faster", "semantic"),
        ("PostgreSQL vacuum freezing best practices", "hybrid-intent"),
        ("K8s pod OOMKilled", "typo/variant"),
    ]

    for query, qtype in queries:
        print(f"Query ({qtype}): '{query}'")

        bm25_results = bm25.search(query, k=3)
        vec_results = vec.search(query, k=3)
        rrf_results = hybrid.search_rrf(query, k=3)
        wlc_results = hybrid.search_wlc(query, k=3, alpha=0.5)

        print(f"  BM25:       {[(d, round(s, 4)) for d, s in bm25_results]}")
        print(f"  Vector:     {[(d, round(s, 4)) for d, s in vec_results]}")
        print(f"  Hybrid RRF: {[(d, round(s, 4)) for d, s in rrf_results]}")
        print(f"  Hybrid WLC: {[(d, round(s, 4)) for d, s in wlc_results]}")
        print()
