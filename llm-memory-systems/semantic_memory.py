"""
Code Block 5: SemanticMemoryStore — vector-based long-term memory with decay.

From: https://dadops.dev/blog/llm-memory-systems/

Stores memories as vectors, retrieves by semantic similarity with
time decay (older memories score lower). Uses seeded random embeddings
to simulate real embeddings for reproducible demos.

Requires: numpy
No API key required.
"""

import numpy as np
from datetime import datetime


class SemanticMemoryStore:
    """Vector-based long-term memory with decay and retrieval."""

    def __init__(self, decay_rate=0.95, embed_dim=64):
        self.memories = []
        self.decay_rate = decay_rate
        self.embed_dim = embed_dim

    def _embed(self, text):
        """Simulate embedding (production: use sentence-transformers or OpenAI)."""
        np.random.seed(hash(text) % 2**31)
        vec = np.random.randn(self.embed_dim)
        return vec / np.linalg.norm(vec)

    def _cosine_sim(self, a, b):
        return float(np.dot(a, b))

    def store(self, text, memory_type="general", importance=0.5, timestamp=None):
        """Store a memory-worthy moment."""
        self.memories.append({
            "text": text,
            "embedding": self._embed(text),
            "type": memory_type,
            "importance": importance,
            "timestamp": timestamp or datetime.now(),
        })

    def retrieve(self, query, top_k=3, now=None):
        """Retrieve the most relevant memories, accounting for decay."""
        if not self.memories:
            return []

        now = now or datetime.now()
        q_embed = self._embed(query)
        scored = []

        for mem in self.memories:
            similarity = self._cosine_sim(q_embed, mem["embedding"])
            days_old = (now - mem["timestamp"]).days
            decay = self.decay_rate ** days_old
            # Final score: semantic similarity * importance * time decay
            score = similarity * mem["importance"] * decay
            scored.append((score, mem))

        scored.sort(key=lambda x: -x[0])
        return [(s, m["text"], m["type"]) for s, m in scored[:top_k]]


if __name__ == "__main__":
    store = SemanticMemoryStore()
    day1 = datetime(2026, 1, 15)
    day30 = datetime(2026, 2, 14)

    # Session 1 (day 1): project setup
    store.store("User prefers pytest with verbose output and coverage reports",
                memory_type="preference", importance=0.8, timestamp=day1)
    store.store("Project uses PostgreSQL 16 with pgvector extension",
                memory_type="decision", importance=0.9, timestamp=day1)

    # Session 2 (day 30): user asks about testing
    results = store.retrieve("How should I set up tests?", top_k=2, now=day30)
    for score, text, mtype in results:
        print(f"[{mtype}] (score: {score:.3f}) {text}")
