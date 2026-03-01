"""
Tiered LLM Cache — exact match, semantic similarity, and prefix-aware caching.

From: https://dadops.dev/blog/llm-cost-optimization/
Code Block 4: "Caching Strategies with Cost Impact"

Blog claims for chatbot (10K convos/day, Sonnet, 4200 in + 350 out):
  - Exact match 25% hit rate → $1,340/month saved
  - Semantic +15% hit rate → $800/month saved
  - Prefix cache → $500/month saved
  - Total: $2,640/month from $5,355 bill
"""

import hashlib
import time
from dataclasses import dataclass


@dataclass
class CacheEntry:
    response: str
    created_at: float
    hit_count: int = 0


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    cost_avoided: float = 0.0
    embedding_cost: float = 0.0


class TieredLLMCache:
    """Three-tier cache: exact match, semantic, prefix-aware."""

    EMBEDDING_COST_PER_CALL = 0.00002  # ~$0.02 per 1K embeddings

    def __init__(self, ttl_seconds: int = 3600,
                 similarity_threshold: float = 0.92):
        self.exact_cache: dict[str, CacheEntry] = {}
        self.semantic_store: list[tuple[list, str, CacheEntry]] = []
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold
        self.stats = CacheStats()

    def _hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _is_fresh(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.created_at) < self.ttl

    def get(self, prompt: str,
            embedding: list[float] | None = None,
            estimated_cost: float = 0.0) -> str | None:
        # Tier 1: exact match
        key = self._hash(prompt)
        if key in self.exact_cache:
            entry = self.exact_cache[key]
            if self._is_fresh(entry):
                entry.hit_count += 1
                self.stats.exact_hits += 1
                self.stats.cost_avoided += estimated_cost
                return entry.response

        # Tier 2: semantic similarity
        if embedding is not None:
            self.stats.embedding_cost += self.EMBEDDING_COST_PER_CALL
            for stored_emb, _, entry in self.semantic_store:
                if self._is_fresh(entry):
                    sim = self._cosine_sim(embedding, stored_emb)
                    if sim >= self.threshold:
                        entry.hit_count += 1
                        self.stats.semantic_hits += 1
                        self.stats.cost_avoided += estimated_cost
                        return entry.response

        self.stats.misses += 1
        return None

    def put(self, prompt: str, response: str,
            embedding: list[float] | None = None):
        entry = CacheEntry(response, time.time())
        self.exact_cache[self._hash(prompt)] = entry
        if embedding is not None:
            self.semantic_store.append((embedding, prompt, entry))

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def report(self) -> str:
        total = (self.stats.exact_hits + self.stats.semantic_hits
                 + self.stats.misses)
        hit_rate = ((self.stats.exact_hits + self.stats.semantic_hits)
                    / total * 100) if total else 0
        net_saved = self.stats.cost_avoided - self.stats.embedding_cost
        return (f"Cache hit rate: {hit_rate:.1f}% "
                f"(exact: {self.stats.exact_hits}, "
                f"semantic: {self.stats.semantic_hits}) | "
                f"Net savings: ${net_saved:.2f}")


if __name__ == "__main__":
    import random

    random.seed(42)
    cache = TieredLLMCache(ttl_seconds=9999)

    # Simulate chatbot scenario: 10K requests/day
    # Blog claims: 25% exact, 15% semantic, 60% miss
    # Per-request cost on Sonnet: (4200 * 3 + 350 * 15) / 1e6 = $0.01785
    per_request_cost = (4200 * 3.0 + 350 * 15.0) / 1e6

    # Create a pool of 100 unique prompts
    prompts = [f"Customer question variant {i}" for i in range(100)]
    # Some prompts are repeated exactly, some are similar
    embeddings = {p: [random.random() for _ in range(8)] for p in prompts}

    # First pass: populate cache with first 50 prompts
    for p in prompts[:50]:
        cache.put(p, f"Response to: {p}", embeddings[p])

    # Second pass: simulate 1000 requests with varying hit patterns
    for i in range(1000):
        if i % 4 == 0:
            # 25% exact match — reuse a cached prompt
            p = prompts[i % 50]
            cache.get(p, estimated_cost=per_request_cost)
        elif i % 4 == 1 and i % 7 != 0:
            # ~15% semantic match — use a slight variation
            p = prompts[(i % 50) + 50] if (i % 50) < 50 else prompts[0]
            # Use same embedding as a cached prompt (simulating high similarity)
            similar_emb = embeddings[prompts[i % 50]]
            cache.get(p, embedding=similar_emb, estimated_cost=per_request_cost)
        else:
            # 60% miss — new prompt
            p = f"Completely new question {i}"
            emb = [random.random() for _ in range(8)]
            cache.get(p, embedding=emb, estimated_cost=per_request_cost)

    print("=== Tiered Cache Simulation ===")
    print(f"  Per-request cost (Sonnet, 4200in+350out): ${per_request_cost:.5f}")
    print(f"  {cache.report()}")
    print(f"\n  Exact hits:    {cache.stats.exact_hits}")
    print(f"  Semantic hits: {cache.stats.semantic_hits}")
    print(f"  Misses:        {cache.stats.misses}")
    print(f"  Cost avoided:  ${cache.stats.cost_avoided:.2f}")
    print(f"  Embedding cost: ${cache.stats.embedding_cost:.4f}")

    # Blog cost claims verification
    print(f"\n=== Blog Claim Verification ===")
    daily_requests = 10_000
    monthly_bill = per_request_cost * daily_requests * 30
    print(f"  Monthly bill (Sonnet, no cache): ${monthly_bill:.0f}")
    print(f"  Blog claims: $5,355/month")

    exact_savings = monthly_bill * 0.25  # 25% exact hit
    semantic_savings = monthly_bill * 0.15  # 15% semantic hit
    prefix_savings = monthly_bill * 0.60 * 0.10  # 60% miss, 10% prefix discount
    print(f"\n  Exact match savings (25%): ${exact_savings:.0f}/month (blog: $1,340)")
    print(f"  Semantic savings (15%):    ${semantic_savings:.0f}/month (blog: $800)")
    print(f"  Prefix cache savings:      ${prefix_savings:.0f}/month (blog: $500)")
    print(f"  Total savings:             ${exact_savings + semantic_savings + prefix_savings:.0f}/month (blog: $2,640)")
