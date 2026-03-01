"""
Three LLM response caching strategies: exact match, semantic, and structural hash.

Blog post: https://dadops.dev/blog/caching-llm-responses/
Code Blocks 2, 3, and 4.

Requires: sentence-transformers, faiss-cpu, numpy.
"""
import hashlib
import sqlite3
import re
import json
import time

import numpy as np


# ── Code Block 2: Exact Match Cache ──

class ExactMatchCache:
    """LLM response cache using SHA-256 hash of normalized prompts.

    Uses SQLite for storage — surprisingly competitive with Redis
    for read-heavy workloads under 1M entries.
    """
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                hash TEXT PRIMARY KEY,
                prompt TEXT,
                response TEXT,
                created_at REAL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)"
        )

    def normalize(self, prompt):
        """Normalize prompt to maximize cache hits."""
        text = prompt.lower().strip()
        text = re.sub(r'\s+', ' ', text)       # collapse whitespace
        text = re.sub(r'[.!?]+$', '', text)     # strip trailing punctuation
        text = re.sub(r'\s*,\s*', ',', text)    # normalize comma spacing
        return text

    def _hash(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, prompt):
        """Look up a cached response. Returns (response, latency_ms) or None."""
        t0 = time.perf_counter()
        normalized = self.normalize(prompt)
        h = self._hash(normalized)
        row = self.conn.execute(
            "SELECT response FROM cache WHERE hash = ?", (h,)
        ).fetchone()
        latency = (time.perf_counter() - t0) * 1000

        if row:
            self.conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1 WHERE hash = ?",
                (h,)
            )
            return row[0], latency
        return None, latency

    def put(self, prompt, response):
        """Store a response in the cache."""
        normalized = self.normalize(prompt)
        h = self._hash(normalized)
        self.conn.execute(
            """INSERT OR REPLACE INTO cache
               (hash, prompt, response, created_at)
               VALUES (?, ?, ?, ?)""",
            (h, prompt, response, time.time())
        )
        self.conn.commit()

    def size(self):
        return self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


# ── Code Block 3: Semantic Cache ──

class SemanticCache:
    """LLM response cache using embedding similarity.

    Embeds queries, stores vectors in a FAISS index, and returns
    cached responses when a similar-enough query is found.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.90):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)  # inner product = cosine on normalized vecs
        self.responses = []       # response text, indexed by position
        self.prompts = []         # original prompts, for debugging

    def _embed(self, text):
        """Embed and L2-normalize for cosine similarity via inner product."""
        vec = self.model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)

    def get(self, prompt):
        """Search for a semantically similar cached query."""
        t0 = time.perf_counter()
        if self.index.ntotal == 0:
            return None, (time.perf_counter() - t0) * 1000

        query_vec = self._embed(prompt)
        scores, indices = self.index.search(query_vec, 1)
        latency = (time.perf_counter() - t0) * 1000

        if scores[0][0] >= self.threshold:
            return self.responses[indices[0][0]], latency
        return None, latency

    def put(self, prompt, response):
        """Add a new entry to the semantic cache."""
        vec = self._embed(prompt)
        self.index.add(vec)
        self.responses.append(response)
        self.prompts.append(prompt)

    def size(self):
        return self.index.ntotal


# ── Code Block 4: Structural Hash Cache ──

# Common English stopwords for normalization
STOPWORDS = frozenset([
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "i", "me", "my", "you", "your", "we", "our", "it", "its",
    "this", "that", "what", "which", "who", "how", "when", "where",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "if", "so", "just", "about",
])


class StructuralHashCache:
    """LLM response cache using structural decomposition.

    Decomposes prompts into components, aggressively normalizes
    each, and hashes the result. Catches structural equivalence
    without embedding overhead.
    """
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                struct_hash TEXT PRIMARY KEY,
                prompt TEXT,
                response TEXT,
                created_at REAL,
                hit_count INTEGER DEFAULT 0
            )
        """)

    def normalize_component(self, text):
        """Aggressive normalization: lowercase, strip stopwords, sort tokens."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)     # remove all punctuation
        text = re.sub(r'\s+', ' ', text)         # collapse whitespace
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOPWORDS]
        tokens.sort()                             # alphabetical sort
        return ' '.join(tokens)

    def decompose(self, prompt):
        """Extract structural components from a prompt.

        Handles three formats:
        - Plain text: treated as a single user component
        - System/User split: if prompt contains a clear separator
        - Template + variables: if prompt has {placeholders}
        """
        components = {}

        # Detect template variables like {context}, {question}
        placeholders = re.findall(r'\{(\w+)\}', prompt)
        if placeholders:
            template = re.sub(r'\{(\w+)\}', '{\\1}', prompt)
            components['template'] = template
            components['vars'] = ','.join(sorted(placeholders))
        else:
            components['body'] = prompt

        return components

    def _struct_hash(self, components):
        """Hash the normalized structural components."""
        parts = []
        for key in sorted(components.keys()):
            normalized = self.normalize_component(components[key])
            parts.append(f"{key}:{normalized}")
        combined = '||'.join(parts)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def get(self, prompt):
        t0 = time.perf_counter()
        components = self.decompose(prompt)
        h = self._struct_hash(components)
        row = self.conn.execute(
            "SELECT response FROM cache WHERE struct_hash = ?", (h,)
        ).fetchone()
        latency = (time.perf_counter() - t0) * 1000

        if row:
            self.conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1 "
                "WHERE struct_hash = ?", (h,)
            )
            return row[0], latency
        return None, latency

    def put(self, prompt, response):
        components = self.decompose(prompt)
        h = self._struct_hash(components)
        self.conn.execute(
            """INSERT OR REPLACE INTO cache
               (struct_hash, prompt, response, created_at)
               VALUES (?, ?, ?, ?)""",
            (h, prompt, response, time.time())
        )
        self.conn.commit()

    def size(self):
        return self.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
