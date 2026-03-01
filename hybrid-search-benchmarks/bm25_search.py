"""BM25 Search using SQLite FTS5

Wraps SQLite's built-in FTS5 full-text search for BM25-ranked keyword search.
No external dependencies — uses Python's built-in sqlite3 module.

Blog post: https://dadops.co/blog/hybrid-search-benchmarks/
Code Block 1 from the blog.
"""

import sqlite3
from typing import List, Tuple


class BM25Search:
    """BM25 search using SQLite FTS5 — fast exact-term matching."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs
            USING fts5(doc_id, content, tokenize='porter')
        """)

    def index(self, documents: List[Tuple[str, str]]):
        """Index a list of (doc_id, content) tuples."""
        self.conn.executemany(
            "INSERT INTO docs (doc_id, content) VALUES (?, ?)",
            documents
        )
        self.conn.commit()

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, bm25_score) pairs. Lower BM25 = better in FTS5."""
        # Quote each term to avoid FTS5 syntax issues (e.g., "-" parsed as NOT)
        terms = query.split()
        fts_query = " ".join('"' + t.replace('"', '""') + '"' for t in terms)
        rows = self.conn.execute("""
            SELECT doc_id, rank FROM docs
            WHERE docs MATCH ? ORDER BY rank LIMIT ?
        """, (fts_query, k)).fetchall()
        return [(row[0], -row[1]) for row in rows]  # negate: higher = better


if __name__ == "__main__":
    # Quick smoke test
    bm25 = BM25Search()

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

    print("=== BM25 Search Tests ===\n")

    # Exact keyword search (BM25 should excel)
    results = bm25.search("error E-4012 connection refused", k=5)
    print("Query: 'error E-4012 connection refused'")
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f}")

    print()

    # Semantic search (BM25 should struggle)
    results = bm25.search("how to make queries faster", k=5)
    print("Query: 'how to make queries faster'")
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f}")

    print()

    # Another keyword search
    results = bm25.search("setMaxPoolSize parameter", k=5)
    print("Query: 'setMaxPoolSize parameter'")
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f}")
